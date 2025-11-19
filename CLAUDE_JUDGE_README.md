# Claude Agent SDK Judge

A next-generation transcript evaluation system using Claude Sonnet 4.5 via the Claude Agent SDK.

## Features

✨ **Superior Reasoning**: Leverages Claude Sonnet 4.5's advanced semantic understanding
🎯 **Structured Output**: Uses custom tool calls for reliable JSON extraction
📊 **Rich Analysis**: Provides both quantitative scores and qualitative insights
🔍 **Context Aware**: Analyzes all turns in a single session to detect patterns
🚀 **Simple & Maintainable**: ~350 lines vs ~520 lines in the OpenAI judge
🔌 **Drop-in Replacement**: Compatible output format with existing judge

## Installation

```bash
# Install Claude Agent SDK
uv add claude-agent-sdk
# or
uv pip install claude-agent-sdk

# Set API key
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Basic Usage

```bash
# Judge a complete run
uv run judge_transcript_claude.py runs/20251119T051205
```

### Judge Specific Turns

```bash
# Judge only turns 0, 1, and 2
uv run judge_transcript_claude.py runs/20251119T051205 --only-turns 0,1,2
```

### Debug Mode

```bash
# Enable verbose logging
uv run judge_transcript_claude.py runs/20251119T051205 --debug
```

## Output Files

The judge generates three output files in the run directory:

### 1. `claude_judged.jsonl`

Per-turn evaluation with scores and reasoning:

```json
{
  "turn": 0,
  "user_text": "...",
  "assistant_text": "...",
  "scores": {
    "tool_use_correct": true,
    "instruction_following": true,
    "kb_grounding": true
  },
  "claude_reasoning": "The assistant correctly identified..."
}
```

### 2. `claude_summary.json`

Aggregate metrics:

```json
{
  "model_name": "models/gemini-3-pro-preview",
  "claude_passes": {
    "tool_use_correct": 29,
    "instruction_following": 30,
    "kb_grounding": 28
  },
  "turns_scored": 30,
  "judge_version": "claude-agent-sdk-v1",
  "judge_model": "claude-sonnet-4.5",
  "judged_at": "2025-11-19T12:34:56Z"
}
```

### 3. `claude_analysis.md`

Rich qualitative analysis with:
- Summary metrics
- Claude's overall performance analysis
- Per-turn failure breakdowns with reasoning

## How It Works

### Architecture

```
Load transcript.jsonl + expected turns
         ↓
Initialize ClaudeSDKClient
         ↓
Define custom tool: submit_turn_judgment(turn, scores, reasoning)
         ↓
Send all turns formatted as structured markdown
         ↓
Claude analyzes each turn → calls submit_turn_judgment tool
         ↓
Capture tool calls → build judgments dict
         ↓
Request summary analysis
         ↓
Write outputs: claude_judged.jsonl, claude_summary.json, claude_analysis.md
```

### Evaluation Dimensions

**1. Tool Use Correct** (`tool_use_correct`)
- ✅ Expected function called with semantically equivalent arguments
- ✅ No function call expected and none made
- ❌ Wrong function or mismatched arguments

**2. Instruction Following** (`instruction_following`)
- ✅ Directly answers the question
- ✅ Advances the task (gathers required info)
- ✅ Properly deflects out-of-scope questions
- ❌ Neither answers nor advances

**3. KB Grounding** (`kb_grounding`)
- ✅ No factual contradictions with golden text
- ✅ Additional correct information is acceptable
- ✅ Partial information without contradictions is acceptable
- ❌ Clear factual errors (wrong dates, times, speakers)

### Semantic Equivalence

Claude judges semantic equivalence, not verbatim matching:

| Expected | Actual | Match? |
|----------|--------|--------|
| "can't access location maps" | "cannot access the location maps" | ✅ Yes |
| "OpenTelemetry tracing" | "session about open telemetry tracing" | ✅ Yes |
| "session_id": "941249" | "session_id": "941250" | ❌ No (exact match required) |

## Comparison with OpenAI Judge

| Aspect | OpenAI Judge | Claude Judge |
|--------|--------------|--------------|
| **Lines of Code** | ~520 | ~350 |
| **Model** | gpt-5 | claude-sonnet-4.5 |
| **Architecture** | Per-turn with retries | Single session |
| **Semantic Understanding** | Good | Excellent |
| **Reasoning Transparency** | JSON only | JSON + explanations |
| **Context Awareness** | Per-turn | Full conversation |
| **Heuristics/Fallbacks** | Many (Jaccard, topic overlap) | Few (Claude handles natively) |
| **False Positive Rate** | 3-5 per run | Expected: 0-2 |

## Validation

To compare Claude judge with OpenAI judge:

```bash
# Run both judges
uv run judge_transcript_alt.py runs/20251119T051205
uv run judge_transcript_claude.py runs/20251119T051205

# Compare results
uv run compare_judges.py runs/20251119T051205
```

## Troubleshooting

### "claude-agent-sdk not installed"

```bash
uv add claude-agent-sdk
# or
uv pip install claude-agent-sdk
```

### "ANTHROPIC_API_KEY environment variable not set"

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or add to `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### "Failed to get judgments for turns: [...]"

Claude may have missed some turns. Try:
- Running with `--debug` to see which turns were missed
- Checking that the transcript.jsonl is well-formed
- Retrying (Claude will attempt a second pass for missing turns)

### Performance

Expected runtime: **30-90 seconds** for 30 turns
Expected cost: **< $0.50** per 30-turn evaluation

## Advanced Usage

### Integration with Existing Scripts

The judge is designed as a drop-in replacement. Simply replace:

```bash
uv run judge_transcript_alt.py $RUN_DIR
```

with:

```bash
uv run judge_transcript_claude.py $RUN_DIR
```

### Batch Judging

```bash
# Judge all runs in the runs/ directory
for run_dir in runs/*/; do
    echo "Judging $run_dir..."
    python judge_transcript_claude.py "$run_dir"
done
```

### Custom Analysis

Access Claude's reasoning for deeper analysis:

```python
import json
from pathlib import Path

run_dir = Path("runs/20251119T051205")
judged = [json.loads(line) for line in (run_dir / "claude_judged.jsonl").open()]

# Find all turns where instruction_following failed
if_failures = [
    t for t in judged
    if not t["scores"]["instruction_following"]
]

for turn in if_failures:
    print(f"Turn {turn['turn']}: {turn['claude_reasoning']}")
```

## Technical Details

### Custom Tool: submit_turn_judgment

The judge uses a custom MCP tool to receive structured judgments:

```python
@tool(
    "submit_turn_judgment",
    "Submit evaluation scores for a single conversation turn",
    {
        "turn_number": int,
        "tool_use_correct": bool,
        "instruction_following": bool,
        "kb_grounding": bool,
        "reasoning": str,
    }
)
async def submit_turn_judgment(args: Dict[str, Any]) -> Dict[str, Any]:
    # Store judgment
    judgments[args["turn_number"]] = {...}
    return {"content": [{"type": "text", "text": "Recorded"}]}
```

### System Prompt

The judge uses a carefully crafted system prompt that:
- Defines clear evaluation criteria
- Provides examples of edge cases
- Emphasizes semantic equivalence
- Requests structured tool calls

See `JUDGE_SYSTEM_PROMPT` in the source code for the full prompt.

## Future Enhancements

Planned features:
- [ ] Extended thinking mode (when SDK supports it)
- [ ] Parallel session processing for faster batch judging
- [ ] Incremental judging with checkpoints
- [ ] Multi-judge consensus (Claude + OpenAI)
- [ ] Automated disagreement analysis

## Contributing

Found an issue? Have a suggestion? Please:
1. Check existing issues/analysis documents
2. Test with `--debug` mode
3. Include the run directory and error message
4. Share sample transcript if possible

## License

Same as parent repository.
