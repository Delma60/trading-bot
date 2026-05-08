# Cognitive ARIA Architecture - Implementation Summary

## Overview
Transformed ARIA from a reactive template-based chatbot to a truly conversational AI agent with persistent contextual memory, emotional awareness, and natural dialogue generation.

## New Components Implemented

### 1. **Working Memory** (`manager/working_memory.py`)
- Tracks the live conversation as a narrative
- Maintains conversation turns with emotion, intent, and action context
- Tracks:
  - Emotional arc of the conversation (user mood evolution)
  - Session tone (neutral, tense, confident, frustrated)
  - Symbols discussed and trades executed
  - Open promises made to the user
  - Inferred and stated user goals

**Key Classes:**
- `ConversationTurn`: Records each exchange with emotion, intent, and outcomes
- `WorkingMemory`: Session-level context that shapes each response

### 2. **Episodic Memory** (`manager/episodic_memory.py`)
- Persists meaningful events across sessions
- Makes the bot feel like it "knows" the user through observation
- Stores:
  - Trades and their outcomes
  - User behaviors and patterns
  - Mistakes and lessons learned
  - Emotional tags on significant events

**Key Methods:**
- `recall_relevant()`: Fetches memories related to current context
- `recall_pattern()`: Detects behavioral patterns (e.g., overtrading)
- `recall_today()`: Gets today's episodes

### 3. **User Model** (`manager/user_model.py`)
- Continuously updated profile of user personality and preferences
- Adapts response style based on observed behavior
- Tracks:
  - Risk appetite (conservative/moderate/aggressive)
  - Trading style and experience level
  - Communication preferences
  - Decision speed and hesitation patterns
  - Trust level in bot (builds/decreases over time)

**Key Methods:**
- `observe()`: Updates model from user behavior
- `get_communication_style()`: Shapes tone and verbosity
- `infer_risk_appetite()`: Deduces from position sizing

### 4. **Inner Monologue** (`manager/inner_monologue.py`)
- ARIA's internal reasoning before responding
- Shapes what gets said, how it gets said, and what's held back
- Generates thoughts in categories:
  - **Observation**: What's happening now
  - **Concern**: Flags before proceeding
  - **Plan**: Directive for response shaping
  - **Question**: Natural follow-ups
  - **Recall**: Relevant memories to mention

**Key Process:**
- Observes current context
- Checks memory for relevant patterns
- Assesses user emotional state
- Forms the most useful response intent
- Suppresses unhelpful content

### 5. **Voice Layer** (`manager/voice_layer.py`)
- Transforms structured agent output into natural, varied speech
- Not a template picker—assembles responses from semantic blocks
- Shaped by:
  - User's emotional state and session tone
  - Communication preferences
  - Inner monologue directives
  - What was just said (avoids repetition)
  - ARIA's personality constraints

**Key Methods:**
- `render()`: Main naturalizer pipeline
- `_adapt_core()`: Shapes output based on directives
- `_maybe_followthrough()`: Adds natural follow-up questions
- `render_greeting()`: References user history naturally

### 6. **Proactive Engine** (`manager/proactive_engine.py`)
- Generates unprompted insights and observations
- Makes the bot feel "alive" rather than purely reactive
- Monitors:
  - Open positions for notable events
  - Session timing (London close, NY open, etc.)
  - Promised alerts firing

**Key Behaviors:**
- Suggests closing positions near profit targets
- Alerts to spreads widening before session close
- Proactively mentions market structure breaks on watched symbols

## Integration into ARIA (`chat.py`)

### Modified `process_message()` Pipeline
The updated pipeline now flows:
```
User Input
    ↓
Pending State Check / Quick Actions
    ↓
Emotion Detection
    ↓
Working Memory: Log User Turn + Track Symbols
    ↓
Intent Classification
    ↓
Agentic Reasoning (unchanged)
    ↓
Inner Monologue: Generate Thoughts
    ↓
Voice Layer: Naturalize Response
    ↓
Working Memory: Log ARIA Turn
    ↓
User Model: Update Based on Interaction
    ↓
Return Naturalized Response
```

### Key Additions to ARIA Class
1. **Initialization** (in `__init__`):
   - Creates all cognitive components
   - Starts proactive engine
   - Observes session start in user model

2. **Emotion Detection** (`_detect_emotion()`):
   - Simple rule-based keywords for frustration, confidence, anxiety, positivity
   - Can be replaced with ML model later

3. **Enhanced Memory Handling**:
   - Working memory tracks conversation narrative
   - Symbols are registered in working memory
   - Conversation turns are logged with emotion and intent

4. **Cleanup** (`shutdown()` and `__del__`):
   - Stops proactive engine, trailing manager, profit guard
   - Ensures graceful shutdown

## Data Flow Examples

### Example 1: User Shows Frustration
```
User: "Why is EURUSD still ranging? This is taking too long."
  ↓ Emotion: frustrated
  ↓ Inner Monologue thinks: "User is stressed. Keep calm. Avoid hedging."
  ↓ Voice Layer adapts: Removes "I think" hedging, makes response direct
  ↓ Episodic Memory: Records user's frustration with EURUSD ranging
  ↓ User Model: Decreases stress threshold by 1
  ↓ Next Session: Remembered that user was frustrated here
```

### Example 2: User Makes a Good Trade
```
User: "Execute BUY on EURUSD like we discussed"
  ↓ Agent executes trade successfully
  ↓ Working Memory: Logs the execution
  ↓ User Model: Records "followed_signal" (trust increases +0.02)
  ↓ Episodic Memory: Stores as positive episode
  ↓ Inner Monologue: Remembers confidence
  ↓ Later: "You've been executing my setups lately — your conviction is good."
```

### Example 3: System Detects Proactive Opportunity
```
Proactive Engine (background):
  ↓ Sees EURUSD position up $22 (close to round number)
  ↓ Checks if it's a discussed symbol
  ↓ Triggers: "EURUSD is up $22 now. Getting close to $20 — worth watching."
```

## Files Modified
- **chat.py**: Integrated cognitive components into ARIA class
  - Added cognitive imports
  - Enhanced `__init__()` to initialize all components
  - Added `_detect_emotion()` method
  - Updated `process_message()` pipeline with cognitive layers
  - Added `shutdown()` method

## Files Created
- `manager/working_memory.py` - Session-level conversation tracking
- `manager/episodic_memory.py` - Persistent event storage
- `manager/user_model.py` - User personality profile
- `manager/inner_monologue.py` - Internal reasoning engine
- `manager/voice_layer.py` - Response naturalization
- `manager/proactive_engine.py` - Unprompted insights

## Persistence
- **User Model**: Saved to `data/user_model.json` on every observe()
- **Episodic Memory**: Saved to `data/episodic_memory.json` on every store()
- **Working Memory**: Session-only (cleared between sessions)

## Future Enhancements
1. ML-based emotion detection (replace keyword rules)
2. Semantic similarity for memory recall (vs. keyword matching)
3. Multi-turn dialogue context compression for long sessions
4. Personality slider for ARIA voice customization
5. Proactive pattern detection (predict next user need before asking)
6. Integration with sentiment analysis on P&L moments
7. Dialogue act classification for more nuanced speech adaptation

## Testing the Integration

### Verify Cognitive Components Activate
```python
# In main.py or interactive session:
from chat import ARIA

aria = ARIA(
    intents_filepath="intents.json",
    broker=broker,
    strategy_manager=sm,
    portfolio_manager=pm,
    risk_manager=rm
)

# Test emotion detection
emotion = aria._detect_emotion("Why is nothing working? This is frustrating.")
assert emotion == "frustrated"

# Test working memory
aria.working_memory.remember_symbol("EURUSD")
assert "EURUSD" in aria.working_memory.symbols_discussed

# Test response naturalization
raw = "You should analyze EURUSD now."
natural = aria.voice_layer.render(raw, [], "analyze_symbol")
# Should be more conversational than raw

# Clean shutdown
aria.shutdown()
```

### Monitor Persistent Learning
```python
# Check user model after several sessions
import json
model = json.loads(open("data/user_model.json").read())
print(f"User sessions: {model['total_sessions']}")
print(f"Trust level: {model['trust_in_bot']}")
print(f"Communication pref: {model['communication_pref']}")

# Check episodic memory
episodes = json.loads(open("data/episodic_memory.json").read())
print(f"Stored episodes: {len(episodes)}")
for ep in episodes[-5:]:  # Last 5
    print(f"  {ep['timestamp']}: {ep['summary']}")
```

## Architecture Benefits

### For Users
1. **Personalized**: Adapts tone based on your observed behavior
2. **Contextual**: Remembers what was discussed across sessions
3. **Proactive**: Alerts you before you ask
4. **Conversational**: Responses feel natural, not templated
5. **Honest**: Flags concerns and hesitations appropriately

### For Developers
1. **Observable**: Internal reasoning (monologue) is logged and inspectable
2. **Learnable**: User model and episodic memory provide feedback loop
3. **Extensible**: New cognitive layers can be added without touching core
4. **Testable**: Each component is independent and mockable
5. **Graceful**: Falls back to quick actions if cognitive layers fail

## Known Limitations & Trade-offs
1. **Simple Emotion Detection**: Rule-based keywords (not ML)
2. **No Dialogue History Compression**: Long sessions keep all turns
3. **No Multi-turn Discourse Planning**: Each response is independent
4. **Proactive Only on Set Timings**: Doesn't truly "listen" to market sentiment
5. **No Counter-factual Reasoning**: Can't simulate "what if" scenarios

## Next Steps for Production
1. Add comprehensive logging of cognitive decisions (for debugging)
2. Implement A/B testing framework for voice layer variations
3. Add metrics dashboard for user model insights
4. Create admin panel to tune cognitive parameters
5. Implement dialogue backup/restore for user privacy
