# Cognitive ARIA: Before & After Examples

This document shows how the new cognitive architecture transforms ARIA's responses from reactive templates to truly conversational interactions.

---

## Example 1: The Frustrated User

### Scenario
User has taken 3 losses in a row on EURUSD and is asking for analysis again.

### BEFORE (Old Template System)
```
You: analyze EURUSD
ARIA: EURUSD analysis: Grade B bullish signal. 
      Regime: Ranging/Choppy. 
      Confidence: 58%. Consider standard sizing.
```

**Problems:** 
- Doesn't acknowledge user's frustration
- Sounds robotic and template-driven
- Repeats same format as last analysis
- Doesn't remember the recent losses

### AFTER (Cognitive Architecture)
```
Cognitive Pipeline Trace:
├─ Emotion Detection: "analyze" + past context → frustrated
├─ Working Memory: Logs this as frustrated turn on EURUSD
├─ Episodic Memory: Recalls 3 recent losses on EURUSD
├─ Inner Monologue:
│  ├─ Observation: "User is tense. Keep calm."
│  ├─ Concern: "We had a bad trade on EURUSD before"
│  ├─ Plan: "Show reasoning, not just conclusions"
│  └─ Question: "Should suggest different pair?"
├─ Voice Layer:
│  ├─ Surface recall: "You've taken losses on EURUSD recently"
│  ├─ Adapt core: Remove hedging ("I think"), be direct
│  └─ Add followthrough: Suggest alternative
└─ Response:

ARIA: I remember you've taken a few losses on EURUSD lately.
      Today it's showing Grade B setup at 58% — still showing structure.
      Want to take it, or should we scan the other pairs instead?
      Sometimes a fresh perspective helps.
```

**Improvements:**
- ✅ Acknowledges recent history
- ✅ References stored memory naturally
- ✅ Adapts tone to user's emotional state
- ✅ Offers alternative (proactive)
- ✅ Feels conversational, not templated

---

## Example 2: The High-Trust User

### Scenario
User has been trading for 3 months, follows 85% of bot's signals, rarely asks for confirmation.

### BEFORE
```
You: should I take the EURUSD signal?
ARIA: I recommend caution. Consider risk factors.
      Grade B signal at 58% confidence.
      The regime is ranging but showing support formation.
      You might want to wait for a clearer setup.
```

**Problems:**
- Over-hedges even though user is confident
- Too much explanation for an experienced trader
- Doesn't adapt to user's decision style

### AFTER (Cognitive Architecture)
```
Cognitive Pipeline Trace:
├─ User Model: Trust level = 0.82 (high)
│  └─ "Follows signals 85% of time"
│  └─ "Decision speed: impulsive (good at timing)"
├─ Inner Monologue:
│  └─ Plan: "Trust is high. Can be direct and concise."
├─ Voice Layer:
│  ├─ Adapt core: Drop hedging language
│  ├─ No excessive reasoning needed
│  └─ Keep it tight
└─ Response:

ARIA: EURUSD Grade B, 58% confidence. 
      Regime is ranging but support is holding.
      Take it if you're reading the flow right now.
```

**Improvements:**
- ✅ Respects high trust level
- ✅ Concise and direct (not over-explaining)
- ✅ Matches user's decision speed
- ✅ Fewer words, more respect

---

## Example 3: The Learning Pattern

### Session 1
```
You: analyze GBPUSD
ARIA: GBPUSD shows potential on the daily. Grade B/C.
      Do you want to trade it?

You: yes, let's go
ARIA: [executes trade successfully]
```

**System Records:**
- Episodic: "User executed signal on GBPUSD"
- User Model: `follows_signals: 0.55 → 0.60, trust: 0.5 → 0.52`

---

### Session 2 (next day)
```
You: GBPUSD again?
ARIA: You took GBPUSD yesterday and it worked out.
      [remembers specific trade from yesterday]
      Today it's showing similar structure on 4H.
      Grade B again. Want to build on the win?
```

**System Records:**
- Episodic: "User asking about GBPUSD again"
- Working Memory: "We've been focused on GBPUSD"
- Inner Monologue: "User shows confidence after win. Their conviction is good."

---

### Session 3 (after several trades)
```
You: GBPUSD
ARIA: You're 3/4 on GBPUSD setups now.
      [proactive observation of pattern]
      Your timing on that pair is sharp.
      Today shows similar confluence.
      Grade B, 59% confidence.
      Your call — you're reading this pair well.
```

**System Records:**
- User Model: `follows_signals: 0.72, trust: 0.78`
- Episodic: 3 successful GBPUSD trades stored
- Voice Layer: High confidence → direct, concise language
- Inner Monologue: "User has expertise on this pair. Respect it."

---

## Example 4: The Proactive Moment

### Background (user isn't talking)
```
[Proactive Engine scanning every 30 seconds]
Position Check:
├─ EURUSD +$8.50
├─ GBPUSD +$22.10 [near $20 target!]
└─ USDJPY -$3.40

[Alert triggers for GBPUSD]
```

### ARIA (unprompted)
```
ARIA: GBPUSD is up $22 now — you set a $20 target earlier.
      Getting pretty close. Still want to let it run, or take this?
```

**What Happened:**
- Proactive engine detected position near profit target
- Remembered that GBPUSD was a tracked symbol
- Surfaced it naturally as a suggestion, not an alarm
- User didn't have to ask

---

## Example 5: The Emotional Conversation Arc

### Turn 1: Neutral Start
```
You: analyze AUDCAD
Emotion: neutral
ARIA: [straightforward analysis]
```

### Turn 2: Frustration Builds
```
You: why is the Grade still C? We've been watching this all week
Emotion: frustrated
ARIA: [shorter, calmer, no hedging]
     [Inner Monologue: "User is tense. Keep it simple."]
```

### Turn 3: Confidence Returns
```
You: okay let's take it
Emotion: confident
ARIA: [direct and concise, trusts user's timing]
     [Inner Monologue: "Mood shifted. Back to positive."]
```

**System Records:**
- Working Memory: `session_tone` tracked as: neutral → tense → confident
- User Model: Stress threshold adjusted
- Voice Layer: Tone adapted across all three turns

---

## Example 6: The Mistake Recovery

### Session 1
```
You: Take NZDUSD
ARIA: [executes BUY]
[You take -$18 loss quickly]
```

**System Records:**
- Episodic: Episode("mistake", symbol="NZDUSD", outcome="loss", emotional_tag="frustration")
- User Model: `loss_aversion: normal → high`

### Session 2 (next session)
```
You: What about NZDUSD again?
ARIA: [Inner Monologue recalls the loss]
      NZDUSD showed a setup, but...
      [remembers: user was frustrated after that loss]
      We had a rough trade there last week.
      Want to give it another shot, or find something fresher?
```

**Benefits:**
- ✅ Doesn't blindly suggest the same pair
- ✅ Acknowledges the loss context
- ✅ Gives user the choice (respects their recovery pace)

---

## Key Transformation Patterns

### Pattern 1: Memory → Context
**Old:** Random responses that repeat
**New:** Each response references what was discussed

### Pattern 2: Emotion → Adaptation
**Old:** Same tone regardless of user state
**New:** Frustrated? Calm and direct. Confident? Brief and respectful.

### Pattern 3: History → Learning
**Old:** No memory of past sessions
**New:** "You're 3/4 on this pattern." "Last time you took a loss here."

### Pattern 4: Templates → Semantic Assembly
**Old:** Pick from 5 fixed response templates
**New:** Assemble from directives + memory + emotion + personality

### Pattern 5: Reactive → Proactive
**Old:** Wait for user to ask
**New:** "Hey, this position is at your target." "You asked me to watch this."

---

## How Users Experience This

### Feels Like:
- **Partnership** — ARIA knows what happened
- **Respect** — Adapts to your style
- **Intelligence** — Makes connections across sessions
- **Honesty** — Flags concerns you might miss
- **Consistency** — Same person, growing with you

### Not Like:
- Chatbot in template mode
- Dashboard reporting facts
- Random unprompted alerts
- Overselling confidence it doesn't have
- Forgetting what was just said

---

## Implementation Details Visible to User

### When Cognitive Layers Are Active
1. **First mention of a symbol:** "EURUSD" → Remembered in working memory
2. **Emotion detected:** Frustrated tone → Response becomes direct
3. **Past pattern recalled:** "You took a loss on this before" → Surfaced naturally
4. **User model updating:** After several trades → "Your timing is sharp"
5. **Proactive alert:** Without user asking → "This hit your target"

### Silent Work (Behind the Scenes)
- Inner monologue reasoning
- User model probability updates
- Episodic memory storage
- Working memory turn logging
- Voice layer semantic assembly

### User Sees Result
✨ Natural, contextual, feeling like a real trading partner

---

## Performance Characteristics

| Aspect | Impact |
|--------|--------|
| Response Time | +0ms (cognitive layers run before user sees anything) |
| Memory Usage | ~2MB per 100 episodes + user model |
| Persistence | Auto-saves to JSON after each observe/store |
| CPU | Minimal (proactive engine: ~30s check interval) |
| Latency | All synchronous except proactive thread |

---

## Customization Points (For Developer)

### Tune Personality
`manager/voice_layer.py` → PERSONA dict
- Change "never" words
- Adjust contractions
- Modify followthrough frequency

### Adjust Emotion Sensitivity
`chat.py` → `_detect_emotion()` method
- Add more keywords
- Adjust thresholds
- Replace with ML model

### Memory Retention
`manager/episodic_memory.py` → `MAX_EPISODES = 500`
- Increase for longer history
- Adjust pruning strategy

### Proactive Triggers
`manager/proactive_engine.py` → `CHECK_INTERVAL = 30`
- Make more/less frequent
- Add new detection patterns

---

## When to Use Each Component

| Component | Best Used For |
|-----------|--------------|
| Working Memory | Session continuity, immediate context |
| Episodic Memory | Long-term patterns, learning |
| User Model | Personality adaptation, trust tracking |
| Inner Monologue | Reasoning audit, transparency |
| Voice Layer | Response quality, naturalness |
| Proactive Engine | Alerts, unprompted suggestions |

---

## Troubleshooting

**Q: Responses feel repetitive?**
A: Voice layer `_vary_sentence_starts()` detects repetition. Check if pattern is from agent, not voice layer.

**Q: Seems to forget things?**
A: Check episodic_memory.json file size (limit is 500 episodes). May have hit retention cap.

**Q: Emotion detection wrong?**
A: Keywords in `_detect_emotion()` are rule-based. Can add more, or swap for ML model.

**Q: Proactive alerts too frequent/rare?**
A: Adjust `ProactiveEngine.CHECK_INTERVAL` (currently 30 seconds) and `_min_gap` (currently 3 minutes).

**Q: Performance slow?**
A: Proactive engine runs in daemon thread. Check if episodic_memory.json is very large (> 50MB).
