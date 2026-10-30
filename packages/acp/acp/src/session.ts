/** One standard ACP session's Agent, configuration, prompt, update, and teardown lifecycle. */

import type { Context } from '@deepseek-ai/cordis'
import {
  RequestError,
  type McpServer,
  type PromptRequest,
  type PromptResponse,
  type SessionConfigOption,
  type SessionNotification,
  type StopReason,
} from '@agentclientprotocol/sdk'
import type {
  Agent,
  AgentHandle,
  AgentOptions,
  AgentStatus,
  AssistantStreamFrame,
} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-working-directory'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import { createUserMessage, errorChain, type UserMessage } from '@deepseek-ai/dsh-llm'
import {
  appendPluginRecord,
  pluginRecordOf,
  type Session,
  type SessionEvent,
  type SessionId,
  type SessionSeq,
  type TurnEndReason,
} from '@deepseek-ai/dsh-session'
import { AcpContentError, admitAcpPrompt } from './content.ts'
import { turnEndToStopReason } from './codec.ts'
import { mountAcpMcpServers } from './mcp.ts'
import { AcpModelControl } from './model-control.ts'
import { assistantUpdates, toolCallUpdate, toolResultUpdate, userUpdates } from './updates.ts'

/** Subagent completion and teardown without depending on the subagent package. */
interface SubagentLifecycle {
  /** Dispose subagent descendants below exact host-owned parents child-first. */
  drainDescendants(parents: readonly Agent[]): Promise<void>
  /** Observe currently managed descendants without closing future admission. */
  waitForChildren(parent: Agent): Promise<boolean>
}

/** Inputs shared by fresh and resumed ACP session construction. */
interface AcpSessionBuildOptions {
  cwd: string
  /** Adopt the ACP cwd for an existing session whose immutable header names another host/path. */
  relocate?: boolean
  mcpServers: readonly McpServer[]
  agentOptions: AgentOptions
  signal: AbortSignal
  notify: (notification: SessionNotification) => Promise<void>
}

/** Fresh ACP session construction inputs. */
export interface CreateAcpSessionOptions extends AcpSessionBuildOptions {
  sessionId: SessionId
}

/** Persisted ACP session construction inputs. */
export interface ResumeAcpSessionOptions extends AcpSessionBuildOptions {
  sessionId: SessionId
}

/** Exact parent prefix used to create one independent native fork. */
export interface ForkAcpSessionOptions extends AcpSessionBuildOptions {
  sessionId: SessionId
  seed: readonly SessionEvent[]
  inheritedEventCount: number
  parentSessionId: SessionId
}

interface StreamedAttempt {
  readonly turn: number
  readonly step: number
  readonly text: Map<number, string>
  readonly reasoning: Map<number, string>
  chunks: number
}

interface ApprovalGrant {
  readonly toolName: string
}

const APPROVAL_GRANT_RECORD = 'plugin:acp/approval-grant'

declare module '@deepseek-ai/dsh-session/types' {
  interface PluginRecordMap {
    'plugin:acp/approval-grant': ApprovalGrant
  }
}

interface InflightPrompt {
  resolve: (reason: StopReason) => void
  reject: (error: Error) => void
  messageId: string | undefined
  messageQueued: boolean
  turn: number | undefined
  endReason: TurnEndReason | undefined
  admissionDone: Promise<void>
  finishAdmission: () => void
  admissionController: AbortController
  cancellationDone: PromiseWithResolvers<void>
  cancelStartedIdle: boolean
  settlementStarted: boolean
  outputError: Error | undefined
  agentError: Error | undefined
}

/** Standard invalid-parameter failure with protocol-safe detail. */
function invalidParams(detail: string): RequestError {
  return RequestError.invalidParams(undefined, detail)
}

/** Standard internal failure with protocol-safe detail. */
function internalError(detail: string): RequestError {
  return RequestError.internalError(undefined, detail)
}

/**
 * Install switch notices and mount the requested MCP clients in the unpublished Agent scope.
 * @param agentCtx - unpublished Agent scope.
 * @param options - workspace and MCP servers requested for the session.
 */
async function composeAgent(agentCtx: Context, options: AcpSessionBuildOptions): Promise<void> {
  installModelSelection(agentCtx)
  await mountAcpMcpServers(agentCtx, options.mcpServers, options.cwd)
}

/**
 * Per-session ACP module. It owns the unpublished Agent composition, model
 * configuration view, one-prompt admission slot, ordered standard updates,
 * and memoized quiescent teardown.
 */
export class AcpSession {
  /** The exact top-level Agent owned by this ACP session. */
  readonly agent: Agent
  private readonly modelControl: AcpModelControl
  private outputTail = Promise.resolve()
  private inflight: InflightPrompt | undefined
  private closing: Promise<void> | undefined

  private constructor(
    private readonly ctx: Context,
    handle: AgentHandle,
    private readonly notify: (notification: SessionNotification) => Promise<void>,
  ) {
    this.agent = handle.agent
    this.modelControl = new AcpModelControl(ctx.llm, handle.agent.model)
    this.disposeAgent = () => handle.dispose()
  }

  private readonly disposeAgent: () => Promise<void>
  private readonly streamedAttempts = new Map<string, StreamedAttempt>()
  private readonly approvalGrants = new Set<string>()

  /**
   * Compose a fresh Agent and all requested MCP clients before publication.
   * @param ctx - ACP plugin context with Agent, LLM, and persistence services.
   * @param options - fresh session identity, workspace, route, MCP, and notifier.
   * @returns the fully composed per-session module.
   */
  static async create(ctx: Context, options: CreateAcpSessionOptions): Promise<AcpSession> {
    const handle = await ctx.agents.create({
      sessionId: options.sessionId,
      meta: { cwd: options.cwd },
      agentOptions: options.agentOptions,
      signal: options.signal,
      setup: agentCtx => composeAgent(agentCtx, options),
    })
    const session = new AcpSession(ctx, handle, options.notify)
    session.hydrateApprovalGrants()
    return session
  }

  /**
   * Compose a fresh Agent over an exact parent prefix.
   * @param ctx - ACP plugin context.
   * @param options - child identity, inherited events, workspace, and MCP servers.
   * @returns the independent child session.
   */
  static async fork(ctx: Context, options: ForkAcpSessionOptions): Promise<AcpSession> {
    const handle = await ctx.agents.create({
      sessionId: options.sessionId,
      seed: options.seed,
      inheritedEventCount: options.inheritedEventCount,
      meta: {
        cwd: options.cwd,
        parentSession: options.parentSessionId,
        isSeeded: true,
      },
      agentOptions: options.agentOptions,
      signal: options.signal,
      setup: agentCtx => composeAgent(agentCtx, options),
    })
    const session = new AcpSession(ctx, handle, options.notify)
    session.hydrateApprovalGrants()
    if (options.relocate === true && !await samePath(session.agent.session.header.cwd, options.cwd)) {
      const workingDirectory = ctx.get('workingDirectory')
      if (workingDirectory === undefined) throw new Error('ACP relocation needs the workingDirectory service')
      await workingDirectory.set(session.agent, options.cwd, options.signal)
    }
    return session
  }

  /**
   * Restore a persisted Agent and compose the request's fresh MCP connections.
   * The Session's recorded model selection survives; deployment options supply
   * the route only when the Session records neither a selection nor a request.
   * @param ctx - ACP plugin context with Agent, LLM, and persistence services.
   * @param options - persisted identity, workspace, initial route, MCP, and notifier.
   * @returns the restored per-session module.
   */
  static async resume(ctx: Context, options: ResumeAcpSessionOptions): Promise<AcpSession> {
    const handle = await ctx.agents.resume({
      resumeSessionId: options.sessionId,
      agentOptions: options.agentOptions,
      signal: options.signal,
      setup: agentCtx => composeAgent(agentCtx, options),
    })
    const session = new AcpSession(ctx, handle, options.notify)
    session.hydrateApprovalGrants()
    if (options.relocate === true && !await samePath(session.agent.session.header.cwd, options.cwd)) {
      const workingDirectory = ctx.get('workingDirectory')
      if (workingDirectory === undefined) throw new Error('ACP relocation needs the workingDirectory service')
      await workingDirectory.set(session.agent, options.cwd, options.signal)
    }
    return session
  }

  /**
   * Whether this module owns an exact Agent reference.
   * @param agent - Agent observed on a scoped runtime event.
   * @returns true only for this session's owned Agent.
   */
  owns(agent: Agent): boolean {
    return this.agent === agent
  }

  /**
   * Whether this module owns an exact Session reference.
   * @param session - Session observed on a durable event.
   * @returns true only for this session's owned Session.
   */
  ownsSession(session: Session): boolean {
    return this.agent.session === session
  }

  /**
   * Return the complete standard model configuration state.
   * @param signal - optional request cancellation.
   * @returns provider-grouped model and exact-model reasoning options.
   */
  configOptions(signal?: AbortSignal): Promise<SessionConfigOption[]> {
    this.assertActive()
    return this.modelControl.options(signal)
  }

  /**
   * Whether this exact session already holds a remembered tool grant.
   * @param toolName - approval request's tool identity.
   */
  hasApprovalGrant(toolName: string): boolean {
    return this.approvalGrants.has(toolName)
  }

  /** Persist one session-scoped approval grant and apply it immediately. */
  rememberApprovalGrant(toolName: string): void {
    if (this.approvalGrants.has(toolName)) return
    this.approvalGrants.add(toolName)
    try {
      appendPluginRecord(this.agent.session, APPROVAL_GRANT_RECORD, { toolName })
    } catch (error: unknown) {
      this.approvalGrants.delete(toolName)
      throw error
    }
  }

  /**
   * Consume one live model-attempt publication. Text and reasoning are sent
   * incrementally; committed tool facts remain the only tool lifecycle source.
   * @param frame - ordered attempt frame owned by this session's Agent.
   */
  onAssistantStream(frame: AssistantStreamFrame): void {
    if (this.closing !== undefined) return
    if (frame.type === 'start') {
      this.streamedAttempts.set(frame.attemptId, {
        turn: frame.turn,
        step: frame.step,
        text: new Map(),
        reasoning: new Map(),
        chunks: 0,
      })
      return
    }
    const attempt = this.streamedAttempts.get(frame.attemptId)
    if (attempt === undefined) return
    if (frame.type === 'end') {
      // A committed assistant/message is consumed by its queued projection,
      // which removes the attempt after delivery. Everything else can never
      // contribute to a later visible message.
      if (frame.outcome.kind === 'abandoned' || frame.outcome.eventType !== 'assistant/message') {
        const committedMessage =
          frame.outcome.kind === 'committed' && frame.outcome.eventType === 'assistant/message'
        if (
          !committedMessage
          && attempt.chunks > 0
          && this.inflight?.admissionController.signal.aborted !== true
        ) {
          this.enqueueUpdate({
            sessionUpdate: 'agent_message_chunk',
            messageId: frame.attemptId,
            content: { type: 'text', text: '' },
            _meta: { dshRetractAttemptId: frame.attemptId },
          })
        }
        this.streamedAttempts.delete(frame.attemptId)
      }
      return
    }
    if (frame.chunk.type !== 'text-delta' && frame.chunk.type !== 'reasoning-delta') return
    const kind = frame.chunk.type === 'text-delta' ? 'text' : 'reasoning'
    const chunks = kind === 'text' ? attempt.text : attempt.reasoning
    const index = frame.chunk.index
    chunks.set(index, (chunks.get(index) ?? '') + frame.chunk.text)
    attempt.chunks += 1
    const update: SessionUpdate = {
      sessionUpdate: kind === 'text' ? 'agent_message_chunk' : 'agent_thought_chunk',
      messageId: frame.attemptId,
      content: { type: 'text', text: frame.chunk.text },
      _meta: {
        dshAttemptId: frame.attemptId,
        dshRevision: frame.revision,
        dshChunkIndex: frame.index,
        dshBlockIndex: frame.chunk.index,
        dshStreamKind: kind,
      },
    }
    this.enqueueUpdate(update)
  }

  /**
   * Replay the committed transcript in durable sequence order. Every
   * notification carries a deterministic replay identity so clients can
   * deduplicate it against their already persisted visible history.
   */
  async replay(): Promise<void> {
    this.assertActive()
    for (const event of this.agent.session.snapshotEvents()) {
      const meta = {
        isReplay: true,
        dshSeq: event.seq,
        dshEventType: event.type,
      }
      const updates: SessionUpdate[] = []
      if (event.type === 'user/message') {
        updates.push(...await userUpdates(this.ctx, event, { ...meta, dshMessageId: event.data.id }))
      } else if (event.type === 'assistant/message') {
        updates.push(...(await assistantUpdates(this.ctx, this.agent.session, event)).map(update => ({
          ...update,
          _meta: { ...meta, dshMessageId: event.data.message.id },
        })))
      } else if (event.type === 'tool/call') {
        updates.push({ ...toolCallUpdate(event), _meta: meta })
      } else if (event.type === 'tool/result') {
        updates.push({ ...await toolResultUpdate(this.ctx, event), _meta: meta })
      }
      for (const update of updates) {
        await this.notify({
          sessionId: this.agent.session.id,
          update,
          _meta: meta,
        })
      }
    }
    await this.outputTail
  }

  /**
   * Apply one standard configuration option from the next model step.
   * @param configId - advertised standard option id.
   * @param value - selected standard option value.
   * @param signal - optional request cancellation.
   * @returns the complete resulting option state.
   */
  setConfig(configId: string, value: unknown, signal?: AbortSignal): Promise<SessionConfigOption[]> {
    this.assertActive()
    return this.modelControl.set(configId, value, signal)
  }

  /** Resolve topology state off-chain, then serialize its notification without blocking execution updates. */
  topologyChanged(): void {
    if (this.closing !== undefined) return
    void this.modelControl.options()
      .then((configOptions) => {
        if (this.closing !== undefined) return
        const previous = this.outputTail
        this.outputTail = previous
          .then(() => this.notify({
            sessionId: this.agent.session.id,
            update: { sessionUpdate: 'config_option_update', configOptions },
          }))
          /* v8 ignore start -- the bridge notifier contains transport failure. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: config-option update failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      })
      /* v8 ignore start -- option discovery contains per-provider failure. */
      .catch((error: unknown) => {
        this.ctx.logger.warn(`acp: config-option update failed: ${errorChain(error)}`)
      })
    /* v8 ignore stop */
  }

  /**
   * Admit one prompt and settle after the root, descendants, and updates finish.
   * @param params - standard ACP prompt request for this session.
   * @param imageEnabled - connection capability advertised at initialization.
   * @param requestSignal - JSON-RPC request cancellation signal.
   * @returns the correlated standard stop reason after ordered updates drain.
   */
  async prompt(
    params: PromptRequest,
    imageEnabled: boolean,
    requestSignal?: AbortSignal,
  ): Promise<PromptResponse> {
    this.assertActive()
    if (this.inflight !== undefined) throw invalidParams('a prompt is already in flight for this session')
    const completion = Promise.withResolvers<StopReason>()
    const admission = Promise.withResolvers<void>()
    const admissionController = new AbortController()
    const inflight: InflightPrompt = {
      resolve: completion.resolve,
      reject: completion.reject,
      messageId: undefined,
      messageQueued: false,
      turn: undefined,
      endReason: undefined,
      admissionDone: admission.promise,
      finishAdmission: admission.resolve,
      admissionController,
      cancellationDone: Promise.withResolvers<void>(),
      cancelStartedIdle: false,
      settlementStarted: false,
      outputError: undefined,
      agentError: undefined,
    }
    this.inflight = inflight
    const onRequestAbort = (): void => { this.cancelPrompt('ACP prompt request cancelled') }
    requestSignal?.addEventListener('abort', onRequestAbort, { once: true })
    /* v8 ignore next -- the SDK dispatches a live signal, then notifies abort through its listener. */
    if (requestSignal?.aborted === true) onRequestAbort()
    try {
      let admissionFailure: unknown
      const promptSelection = this.agent.model.current()
      try {
        if (this.ctx.agents.get(this.agent.id) !== this.agent) {
          throw internalError('prompt was not queued: the agent was disposed outside the bridge')
        }
        const content = await admitAcpPrompt(
          this.ctx,
          promptSelection,
          params.prompt,
          imageEnabled,
          admissionController.signal,
        )
        admissionController.signal.throwIfAborted()
        if (this.ctx.agents.get(this.agent.id) !== this.agent) {
          throw internalError('prompt was not queued: the agent was disposed outside the bridge')
        }
        const message = createUserMessage({
          content,
          source: { kind: 'user' },
        })
        inflight.messageId = message.id
        inflight.messageQueued = true
        try {
          this.agent.followup(message)
        } catch (error: unknown) {
          inflight.messageQueued = false
          throw error
        }
      } catch (error: unknown) {
        admissionFailure = error
      } finally {
        inflight.finishAdmission()
      }

      if (inflight.admissionController.signal.aborted) {
        this.settleAfterQuiescence(inflight)
        return { stopReason: await completion.promise }
      }
      if (admissionFailure !== undefined) {
        this.inflight = undefined
        if (admissionFailure instanceof AcpContentError) {
          throw admissionFailure.kind === 'invalid'
            ? invalidParams(admissionFailure.message)
            : internalError(admissionFailure.message)
        }
        if (admissionFailure instanceof RequestError) throw admissionFailure
        throw internalError(`prompt was not queued: ${(admissionFailure as Error).message}`)
      }

      this.settleAfterQuiescence(inflight)
      return { stopReason: await completion.promise }
    } finally {
      requestSignal?.removeEventListener('abort', onRequestAbort)
    }
  }

  /** Cancel the active prompt, or autonomous work when no ACP prompt exists. */
  cancel(): void {
    const inflight = this.inflight
    this.cancelPrompt('ACP prompt cancelled')
    if (inflight === undefined) this.agent.cancel({ kind: 'user' })
  }

  /**
   * Process one durable event and enqueue its standard ACP projections.
   * @param session - exact event-owning Session.
   * @param event - committed durable event.
   */
  onSessionEvent(session: Session, event: SessionEvent): void {
    if (event.type === 'turn/start' && this.inflight?.turn !== undefined) {
      this.inflight.turn = event.data.turn
      this.inflight.endReason = undefined
    }
    try {
      if (event.type === 'assistant/message') {
        const inflight = this.inflight?.turn === event.data.turn ? this.inflight : undefined
        const streamed = this.streamedPayload(event.data.turn, event.data.step)
        const previous = this.outputTail
        const delivery = previous.then(async () => {
          for (const update of await assistantUpdates(this.ctx, session, event, streamed)) {
            await this.notify({ sessionId: this.agent.session.id, update })
          }
        })
        this.outputTail = delivery
          .catch((error: unknown) => {
            const failure = error as Error
            if (inflight !== undefined) inflight.outputError ??= failure
            this.ctx.logger.warn(`acp: assistant output conversion failed: ${errorChain(error)}`)
          })
          .finally(() => {
            this.clearStreamedPayload(event.data.turn, event.data.step)
          })
      } else if (event.type === 'tool/call') {
        const previous = this.outputTail
        this.outputTail = previous
          .then(() => this.notify({ sessionId: this.agent.session.id, update: toolCallUpdate(event) }))
          /* v8 ignore start -- the bridge notifier contains transport rejection. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: tool-call update delivery failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      } else if (event.type === 'tool/result') {
        const previous = this.outputTail
        this.outputTail = previous
          .then(async () => this.notify({
            sessionId: this.agent.session.id,
            update: await toolResultUpdate(this.ctx, event),
          }))
          /* v8 ignore start -- supplemental-content conversion failure is contained and cannot fail Agent work. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: tool-result update delivery failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      }
    } finally {
      const inflight = this.inflight
      if (inflight !== undefined && event.type === 'turn/end' && inflight.turn === event.data.turn) {
        inflight.endReason = event.data.reason
        inflight.agentError = undefined
      }
    }
  }

  /**
   * Correlate an accepted user message with its Agent turn.
   * @param message - claimed durable inbox message.
   * @param turn - allocated Agent turn.
   */
  onInboxClaimed(message: UserMessage, turn: number): void {
    if (this.inflight !== undefined && this.inflight.messageId === message.id) this.inflight.turn = turn
  }

  /**
   * Correlate an Agent interval failure with the active ACP prompt.
   * @param error - original same-process failure.
   */
  onAgentError(error: unknown): void {
    const inflight = this.inflight
    if (inflight === undefined || !inflight.messageQueued) return
    // A later committed turn/end supersedes this live failure. Failed turn
    // start/end appends have no terminal record and keep the failure observable.
    inflight.agentError = new Error(errorChain(error))
    this.settleAfterQuiescence(inflight)
  }

  /**
   * Complete cancelled activity without following a later root driver.
   * @param status - the emitted transition, which may precede a reentrant wake.
   */
  onAgentStatus(status: AgentStatus): void {
    const inflight = this.inflight
    if (inflight === undefined || !inflight.messageQueued || !inflight.admissionController.signal.aborted) return
    // Maintenance remains publicly idle; its next running transition belongs
    // to work released after the maintenance task finished.
    if (status === 'idle' || inflight.cancelStartedIdle) inflight.cancellationDone.resolve(this.outputTail)
  }

  /** Await every update queued before this call. */
  drainUpdates(): Promise<void> {
    return this.outputTail
  }

  /**
   * Cancel, drain, flush, and dispose this session once.
   * @param detail - cancellation detail for any prompt still in admission.
   * @returns the shared quiescent teardown promise.
   */
  close(detail: string): Promise<void> {
    if (this.closing !== undefined) return this.closing
    this.closing = (async () => {
      const failures: unknown[] = []
      const inflight = this.inflight
      this.cancelPrompt(detail)
      if (inflight === undefined || !inflight.messageQueued) this.agent.cancel({ kind: 'user' })
      try {
        await inflight?.admissionDone
        await this.agent.whenIdle()
        await this.outputTail
      } catch (error: unknown) {
        failures.push(new Error('ACP session activity drain failed', { cause: error }))
      }
      const subagents = this.ctx.get('subagents') as SubagentLifecycle | undefined
      try {
        await subagents?.drainDescendants([this.agent])
      } catch (error: unknown) {
        this.ctx.logger.warn(`acp: subagent teardown failed: ${errorChain(error)}`)
        failures.push(new Error('subagent teardown failed', { cause: error }))
      }
      try {
        await this.ctx.sessions.flush(this.agent.session)
      } catch (error: unknown) {
        failures.push(new Error('ACP session persistence flush failed', { cause: error }))
      }
      try {
        await this.disposeAgent()
      } catch (error: unknown) {
        failures.push(error)
      }
      if (failures.length === 1) throw failures[0]
      /* v8 ignore start -- independent teardown failures can aggregate only under multiple simultaneous provider faults. */
      if (failures.length > 1) {
        throw new AggregateError(failures, `ACP session teardown failed: ${failures.map(errorChain).join('; ')}`)
      }
      /* v8 ignore stop */
    })()
    return this.closing
  }

  /** Queue one live notification behind all previously emitted protocol output. */
  private enqueueUpdate(update: SessionUpdate): void {
    const previous = this.outputTail
    this.outputTail = previous
      .then(() => this.notify({ sessionId: this.agent.session.id, update }))
      /* v8 ignore start -- the bridge notifier contains transport failures. */
      .catch((error: unknown) => {
        this.ctx.logger.warn(`acp: assistant stream delivery failed: ${errorChain(error)}`)
      })
    /* v8 ignore stop */
  }

  /** Read the exact text and reasoning already emitted for one turn step. */
  private streamedPayload(
    turn: number,
    step: number,
  ): {
    readonly streamedText?: ReadonlyMap<number, string>
    readonly streamedReasoning?: ReadonlyMap<number, string>
  } {
    const matching = [...this.streamedAttempts.values()].filter(
      attempt => attempt.turn === turn && attempt.step === step && attempt.chunks > 0,
    )
    if (matching.length === 0) return {}
    // Retried attempts settle and leave this map before the next attempt
    // starts. The last remaining attempt is the one that committed the
    // visible assistant message, including multi-block responses.
    const attempt = matching.at(-1)!
    return {
      ...attempt.text.size > 0 ? { streamedText: attempt.text } : {},
      ...attempt.reasoning.size > 0 ? { streamedReasoning: attempt.reasoning } : {},
    }
  }

  /** Release transient attempt text after its durable message has settled. */
  private clearStreamedPayload(turn: number, step: number): void {
    for (const [id, attempt] of this.streamedAttempts) {
      if (attempt.turn === turn && attempt.step === step) this.streamedAttempts.delete(id)
    }
  }

  /** Restore session-scoped remembered grants from the durable log. */
  private hydrateApprovalGrants(): void {
    for (const event of this.agent.session.snapshotEvents()) {
      const record = pluginRecordOf(event)
      if (record?.type !== APPROVAL_GRANT_RECORD) continue
      const data: unknown = record.data
      if (
        typeof data === 'object'
        && data !== null
        && 'toolName' in data
        && typeof data.toolName === 'string'
        && data.toolName.length > 0
      ) this.approvalGrants.add(data.toolName)
    }
  }

  private assertActive(): void {
    if (this.closing !== undefined) throw invalidParams(`session is closing: ${this.agent.session.id}`)
  }

  private cancelPrompt(detail: string): void {
    const inflight = this.inflight
    if (inflight === undefined) return
    const firstCancellation = !inflight.admissionController.signal.aborted
    if (firstCancellation) inflight.cancelStartedIdle = this.agent.status === 'idle'
    inflight.admissionController.abort(new Error(detail))
    this.settleAfterQuiescence(inflight)
    if (!inflight.messageQueued) {
      inflight.cancellationDone.resolve()
      return
    }
    this.agent.cancel({ kind: 'user' })
    if (firstCancellation) {
      const done = inflight.cancellationDone
      // Plugin disposal can detach status observers before this activity retires.
      void this.agent.whenIdle().then(() => { done.resolve(this.outputTail) }, done.reject)
    }
  }

  private settleAfterQuiescence(inflight: InflightPrompt): void {
    if (inflight.settlementStarted) return
    inflight.settlementStarted = true
    void (async () => {
      await inflight.admissionDone
      if (inflight.messageQueued) {
        const subagents = this.ctx.get('subagents') as SubagentLifecycle | undefined
        const cancelled = inflight.cancellationDone.promise
        const isCancelled = (): boolean => inflight.admissionController.signal.aborted
        const idleAt = (seq: Session['seq']): boolean => this.agent.status === 'idle' && this.agent.session.seq === seq
        while (true) {
          await Promise.race([this.agent.whenIdle(), cancelled])
          if (isCancelled()) break
          const idleSeq = this.agent.session.seq
          const children = await Promise.race([subagents?.waitForChildren(this.agent), cancelled])
          if (isCancelled()) break
          if (children || !idleAt(idleSeq)) continue
          await this.outputTail
          if (isCancelled() || idleAt(idleSeq)) break
        }
        if (isCancelled()) await cancelled
      }
      /* v8 ignore next -- this prompt owns the slot until this exact settlement clears it. */
      if (this.inflight !== inflight) return
      this.inflight = undefined
      if (inflight.admissionController.signal.aborted) {
        inflight.resolve('cancelled')
        return
      }
      if (inflight.outputError !== undefined) {
        inflight.reject(internalError(`assistant output delivery failed: ${inflight.outputError.message}`))
        return
      }
      if (inflight.agentError !== undefined) {
        inflight.reject(internalError(`turn failed: ${inflight.agentError.message}`))
        return
      }
      const end = inflight.endReason
      if (end === undefined) {
        inflight.resolve('cancelled')
      } else if (end.kind === 'error') {
        inflight.reject(internalError(`turn failed: ${end.error.message}`))
      } else {
        inflight.resolve(turnEndToStopReason(end))
      }
    })()
      /* v8 ignore start -- admissionDone only resolves; idle/output gates contain their own failures. */
      .catch((error: unknown) => {
        if (this.inflight !== inflight) return
        this.inflight = undefined
        inflight.reject(internalError(`prompt settlement failed: ${errorChain(error)}`))
      })
    /* v8 ignore stop */
  }
}

/** Compare paths after the fs service has already validated the destination. */
async function samePath(left: string | undefined, right: string): Promise<boolean> {
  if (left === undefined) return false
  const { realpath } = await import('node:fs/promises')
  const { resolve } = await import('node:path')
  try {
    const [a, b] = await Promise.all([realpath(left), realpath(right)])
    return a === b
  } catch (_unresolvablePath) {
    return resolve(left) === resolve(right)
  }
}
