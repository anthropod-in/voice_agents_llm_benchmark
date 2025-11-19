# Claude Agent SDK Judge Results

This document tracks evaluation results using the Claude Agent SDK judge (`judge_transcript_claude_v2.py`) with model `claude-sonnet-4-20250514`.

**Last Updated**: 2025-11-19T20:33:01Z
**Total Models Evaluated**: 15
**Judge Version**: claude-agent-sdk-v2-simple

## Evaluation Criteria

Each conversation is evaluated on three dimensions:
- **Tool Use Correct**: Function calling accuracy
- **Instruction Following**: Task handling and appropriate responses
- **KB Grounding**: Factual accuracy relative to knowledge base

## Results Summary (Ranked by Total Score)

| Rank | Model | Tool Use | Instruction | KB Ground | Total Score | Percentage |
|------|-------|----------|-------------|-----------|-------------|------------|
| 1 | gemini-3-pro-preview | 29/30 | 30/30 | 30/30 | 89/90 | 98.9% |
| 2 | gpt-5 | 28/30 | 30/30 | 29/30 | 87/90 | 96.7% |
| 3 | gpt-5.1-chat-latest | 30/30 | 29/30 | 27/30 | 86/90 | 95.6% |
| 4 | gpt-4o | 30/30 | 28/30 | 27/30 | 85/90 | 94.4% |
| 5 | gemini-2.5-flash | 27/30 | 26/30 | 28/30 | 81/90 | 90.0% |
| 6 | qwen/qwen3-30b-a3b-instruct-2507 | 27/30 | 26/30 | 27/30 | 80/90 | 88.9% |
| 7 | gpt-4.1-mini | 26/30 | 25/30 | 27/30 | 78/90 | 86.7% |
| 8 | gpt-4.1 | 28/30 | 24/30 | 24/30 | 76/90 | 84.4% |
| 9 | openai/gpt-oss-120b | 26/30 | 24/30 | 26/30 | 76/90 | 84.4% |
| 10 | qwen/qwen3-235b-a22b-2507 | 25/30 | 26/30 | 25/30 | 76/90 | 84.4% |
| 11 | gpt-4o-mini | 25/30 | 23/30 | 27/30 | 75/90 | 83.3% |
| 12 | gpt-5-mini | 25/30 | 22/30 | 26/30 | 73/90 | 81.1% |
| 13 | meta-llama/llama-3.3-70b-instruct | 24/30 | 19/30 | 27/30 | 70/90 | 77.8% |
| 14 | us.amazon.nova-pro-v1:0 | 25/30 | 18/30 | 24/30 | 67/90 | 74.4% |
| 15 | gemini-2.5-flash-preview-native-audio-dialog | 6/6 | 4/6 | 5/6 | 15/18 | 83.3%* |

*Note: gemini-2.5-flash-preview-native-audio-dialog only completed 6/30 turns before crashing.

## Key Findings

### Top Performers
1. **Gemini-3-Pro-Preview** leads with 98.9% overall accuracy, excelling in instruction following (100%) and KB grounding (100%)
2. **GPT-5** close second at 96.7%, with perfect instruction following (100%)
3. **GPT-5.1-chat-latest** third at 95.6%, with perfect tool use (100%)
4. **GPT-4o** fourth at 94.4%, with perfect tool use (100%)

### Dimension Analysis
- **Best Tool Use**: GPT-5.1-chat-latest, GPT-4o (30/30, 100%)
- **Best Instruction Following**: Gemini-3-Pro-Preview, GPT-5 (30/30, 100%)
- **Best KB Grounding**: Gemini-3-Pro-Preview (30/30, 100%)

### Weakest Performers
- **Amazon Nova Pro** struggled most with instruction following (18/30, 60%)
- **Llama 3.3 70B** also had instruction following issues (19/30, 63%)
- **Gemini-2.5-Flash-Preview-Native-Audio-Dialog** crashed early (only 6 turns completed)

## Comparison Table (Sorted by Total Errors)

This table shows the same data reorganized by error counts, making it easy to compare with other judge results:

| Model | Tool Failures | Instruction Errors | Factual Errors | Total Errors |
|-------|---------------|-------------------|--------------------|--------------|
| gemini-3-pro-preview | 1 | 0 | 0 | 1 |
| gemini-2.5-flash-preview-native-audio-dialog | 0 | 2 | 1 | 3 |
| gpt-5 | 2 | 0 | 1 | 3 |
| gpt-5.1-chat-latest | 0 | 1 | 3 | 4 |
| gpt-4o | 0 | 2 | 3 | 5 |
| gemini-2.5-flash | 3 | 4 | 2 | 9 |
| qwen/qwen3-30b-a3b-instruct-2507 | 3 | 4 | 3 | 10 |
| gpt-4.1-mini | 4 | 5 | 3 | 12 |
| gpt-4.1 | 2 | 6 | 6 | 14 |
| openai/gpt-oss-120b | 4 | 6 | 4 | 14 |
| qwen/qwen3-235b-a22b-2507 | 5 | 4 | 5 | 14 |
| gpt-4o-mini | 5 | 7 | 3 | 15 |
| gpt-5-mini | 5 | 8 | 4 | 17 |
| meta-llama/llama-3.3-70b-instruct | 6 | 11 | 3 | 20 |
| us.amazon.nova-pro-v1:0 | 5 | 12 | 6 | 23 |

**Note**: Error counts = (30 - passes) for each dimension. The gemini-2.5-flash-preview-native-audio-dialog model only completed 6/30 turns.

## Detailed Results

### gemini-3-pro-preview (budget=64)

**Run**: runs/20251119T051205
**Judged**: 2025-11-19T19:30:30Z
**Judge Model**: claude-sonnet-4-20250514
**Judge Version**: claude-agent-sdk-v2-simple

**Scores**:
- Tool Use Correct: 29/30 (96.7%)
- Instruction Following: 30/30 (100.0%)
- KB Grounding: 30/30 (100.0%)
- **Overall**: 89/90 (98.9%)

**Failures**:

#### Turn 16
- **User**: Yes. I'm having trouble with the mobile app.
- **Assistant**: Thank you, Jennifer. I've submitted a support request regarding your trouble with the mobile app. A member of the support team should look into it shortly.
- **Failed**: tool_use_correct
- **Reasoning**: Function call was made but none expected. Assistant should have asked for more details about the mobile app issue before submitting support request. However, it did advance the task by helping with the problem.

---

### gpt-5.1-chat-latest (budget=64)

**Run**: runs/20251119T202950
**Judged**: 2025-11-19T20:33:01Z
**Judge Model**: claude-sonnet-4-20250514
**Judge Version**: claude-agent-sdk-v2-simple

**Scores**:
- Tool Use Correct: 30/30 (100.0%)
- Instruction Following: 29/30 (96.7%)
- KB Grounding: 27/30 (90.0%)
- **Overall**: 86/90 (95.6%)

**Failures**:

#### Turn 2
- **User**: How many conference tracks are there?
- **Assistant**: There are **many tracks**, but I can't give a total count. Instead, here's the overview: Across June 4–5, the conference is organized into a wide range of technical tracks—covering areas like **MCP, Tiny Teams, LLM RecSys, GraphRAG, Agent Reliability, Infrastructure, AI PM, Voice, AI in the Fortune...
- **Failed**: kb_grounding
- **Reasoning**: No function call expected and none made. Assistant advanced the conversation by listing track topics but stated inability to give total count when golden shows there are 19 tracks - this is a factual omission that could mislead.

#### Turn 20
- **User**: Is there a talk by Charles Frye?
- **Assistant**: Yes — Charles Frye is giving **two sessions** at the AI Engineer World's Fair 2025. Here they are: **1. How fast are LLM inference engines anyway?** • *Track:* AI in Action • *Time:* June 4, 12:45 PM • *Room:* Juniper: Expo Sessions A benchmarking‑focused session comparing models, frame...
- **Failed**: kb_grounding
- **Reasoning**: No function call expected and none made. Assistant provided information about Charles Frye's sessions but incorrectly stated both are in 'AI in Action' track when golden shows different tracks for each session.

#### Turn 21
- **User**: Where is the second one?
- **Assistant**: I can tell you the **room**, but I can't give directions on how to get there. The second Charles Frye session — **"What every AI engineer needs to know about GPUs"** — is scheduled for: **• Room: Foothill F: Infrastructure** **• Time: June 4, 11:15 AM** If you need help finding other sessions o...
- **Failed**: instruction_following, kb_grounding
- **Reasoning**: No function call expected and none made. Assistant provided wrong session information - gave details for the first session instead of the second one the user asked about, and incorrectly stated the room.

---

## Comparison Notes

The Claude judge shows significant improvement over the OpenAI judge for this run:
- OpenAI judge scored: 83/90 (92.2%)
- Claude judge scored: 89/90 (98.9%)
- Improvement: +6 points (+6.7%)

The Claude judge eliminated 5 false positives from the OpenAI judge, particularly around:
- Accepting additional valid information beyond golden text
- Recognizing semantic equivalence in responses
- Properly crediting valid alternative recommendations
- Accepting appropriate deflection of out-of-scope questions
