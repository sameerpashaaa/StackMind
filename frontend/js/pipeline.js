/**
 * StackMind — Multi-Agent Pipeline Orchestrator
 *
 * Manages pipelineState and drives the 5-stage pipeline execution.
 * Each mutation calls onStateChange(newState) so the view can re-render.
 *
 * Stage 1 → capture query
 * Stage 2 → decompose (LLM)
 * Stage 3 → 3 agents in parallel (Promise.all)
 * Stage 4 → synthesize compilation
 * Stage 5 → display final answer
 *
 * "Regenerate" (stages 4-5 only) re-uses the existing agent outputs.
 */

import { api } from './api.js';

/** @returns {Object} A fresh initial pipeline state */
export function createInitialPipelineState() {
  return {
    status: 'idle',          // 'idle' | 'running' | 'done' | 'error'
    query: '',               // Stage 1: raw user query
    inputType: 'text',       // 'text' | 'image' | 'code' | 'voice'
    decomposition: null,     // Stage 2: { summary, agents: [{id, task}] }
    agents: [                // Stage 3
      { id: 1, task: '', status: 'pending', output: '' },
      { id: 2, task: '', status: 'pending', output: '' },
      { id: 3, task: '', status: 'pending', output: '' },
    ],
    compilation: { status: 'pending' },  // Stage 4
    finalAnswer: '',                     // Stage 5
    error: null,
    conversationContext: null, // For follow-up: { query, answer }
  };
}

/**
 * Run the full 5-stage pipeline.
 *
 * @param {string} query          - The user's raw input text
 * @param {string} inputType      - Input mode: 'text' | 'image' | 'code' | 'voice'
 * @param {Object|null} context   - Prior conversation context { query, answer } or null
 * @param {Function} onStateChange - Called with the full new state after each mutation
 */
export async function runPipeline(query, inputType, context, onStateChange) {
  let state = {
    ...createInitialPipelineState(),
    status: 'running',
    query,
    inputType,
    conversationContext: context || null,
  };

  const emit = (patch) => {
    state = _merge(state, patch);
    onStateChange({ ...state });
  };

  try {
    // ── Stage 2: Decompose ──────────────────────────────────────
    const contextStr = context
      ? `Previous question: "${context.query}"\nPrevious answer summary: "${context.answer.slice(0, 300)}"`
      : null;

    const decomposition = await api.pipelineDecompose(query, contextStr);

    emit({
      decomposition,
      agents: decomposition.agents.map((a) => ({
        id: a.id,
        task: a.task,
        status: 'pending',
        output: '',
      })),
    });

    // ── Stage 3: Parallel agent execution ──────────────────────
    // Mark all three agents as active simultaneously
    emit({
      agents: state.agents.map((a) => ({ ...a, status: 'active' })),
    });

    const agentResults = await Promise.all(
      state.agents.map(async (agent) => {
        const result = await api.pipelineRunAgent(agent.task, agent.id, query);
        // Emit partial update as each agent finishes
        emit({
          agents: state.agents.map((a) =>
            a.id === agent.id
              ? { ...a, status: 'done', output: result.output }
              : a
          ),
        });
        return { id: agent.id, task: agent.task, output: result.output };
      })
    );

    // ── Stage 4: Compilation ────────────────────────────────────
    emit({ compilation: { status: 'active' } });

    const synthesis = await api.pipelineSynthesize(query, agentResults);

    emit({ compilation: { status: 'done' } });

    // ── Stage 5: Final answer ───────────────────────────────────
    emit({
      status: 'done',
      finalAnswer: synthesis.final_answer,
    });

  } catch (err) {
    emit({
      status: 'error',
      error: err.message || 'An unexpected error occurred.',
    });
  }
}

/**
 * Re-run only stages 4-5 (Regenerate) using the existing agent outputs.
 * Agent outputs from state.agents are reused without re-calling the LLM.
 *
 * @param {Object}   currentState  - Current pipeline state (must be 'done')
 * @param {Function} onStateChange - Same callback as runPipeline
 */
export async function regeneratePipeline(currentState, onStateChange) {
  let state = {
    ...currentState,
    status: 'running',
    compilation: { status: 'active' },
    finalAnswer: '',
  };

  const emit = (patch) => {
    state = _merge(state, patch);
    onStateChange({ ...state });
  };

  emit({}); // trigger initial re-render with compilation active

  try {
    const agentPayloads = state.agents.map((a) => ({
      id: a.id,
      task: a.task,
      output: a.output,
    }));

    const synthesis = await api.pipelineSynthesize(state.query, agentPayloads);

    emit({ compilation: { status: 'done' } });
    emit({
      status: 'done',
      finalAnswer: synthesis.final_answer,
    });
  } catch (err) {
    emit({
      status: 'error',
      compilation: { status: 'done' },
      error: err.message || 'Regeneration failed.',
    });
  }
}

/** Shallow-merge helper that handles nested arrays properly */
function _merge(state, patch) {
  const merged = { ...state, ...patch };
  // If patch has agents array, do a per-item merge to handle partial updates
  if (patch.agents) {
    merged.agents = patch.agents.map((newAgent) => {
      const existing = state.agents.find((a) => a.id === newAgent.id);
      return existing ? { ...existing, ...newAgent } : newAgent;
    });
  }
  return merged;
}
