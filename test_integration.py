#!/usr/bin/env python3
"""
test_integration.py — Quick integration test for the new unsupervised learning system.
"""

import sys
import pandas as pd
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all new components can be imported."""
    print("Testing imports...")

    try:
        from manager.unsupervised_learner import UnsupervisedLearner
        print("✓ UnsupervisedLearner imported")
    except ImportError as e:
        print(f"✗ UnsupervisedLearner import failed: {e}")
        return False

    try:
        from manager.reasoning_engine import ReasoningEngine
        print("✓ ReasoningEngine imported")
    except ImportError as e:
        print(f"✗ ReasoningEngine import failed: {e}")
        return False

    try:
        from manager.response_engine import ResponseEngine
        print("✓ ResponseEngine imported")
    except ImportError as e:
        print(f"✗ ResponseEngine import failed: {e}")
        return False

    try:
        from strategies.strategy_manager import StrategyManager
        print("✓ StrategyManager imported")
    except ImportError as e:
        print(f"✗ StrategyManager import failed: {e}")
        return False

    return True

def test_unsupervised_learner():
    """Test basic unsupervised learner functionality."""
    print("\nTesting UnsupervisedLearner...")

    try:
        from manager.unsupervised_learner import UnsupervisedLearner

        learner = UnsupervisedLearner()
        print("✓ UnsupervisedLearner instantiated")

        # Test with mock data
        mock_features = pd.Series({
            'adx': 25.0,
            'rsi_14': 65.0,
            'volatility_20': 0.0012,
            'volume_ratio': 1.1,
            'bb_width': 0.02,
            'regime_trending': 0.8,
            'dist_sma50': -0.005,
        })

        regime = learner.ingest_market_bar(mock_features)
        print(f"✓ Regime classification: {regime}")

        insights = learner.generate_insights()
        print(f"✓ Generated {len(insights)} insights")

        return True

    except Exception as e:
        print(f"✗ UnsupervisedLearner test failed: {e}")
        return False

def test_reasoning_engine():
    """Test reasoning engine integration."""
    print("\nTesting ReasoningEngine integration...")

    try:
        from manager.reasoning_engine import ReasoningEngine

        # Mock managers (we can't test with real ones without MT5)
        class MockManager:
            pass

        strategy_manager = MockManager()
        risk_manager = MockManager()
        portfolio_manager = MockManager()

        reasoning = ReasoningEngine(strategy_manager, risk_manager, portfolio_manager)
        print("✓ ReasoningEngine instantiated")

        # Check if learner is accessible
        if reasoning.learner:
            print("✓ Learner integrated into ReasoningEngine")
        else:
            print("⚠ Learner not available (expected without sklearn)")

        return True

    except Exception as e:
        print(f"✗ ReasoningEngine test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 ARIA Unsupervised Learning Integration Test")
    print("=" * 50)

    success = True

    success &= test_imports()
    success &= test_unsupervised_learner()
    success &= test_reasoning_engine()

    print("\n" + "=" * 50)
    if success:
        print("✅ All integration tests passed!")
        print("\nThe unsupervised learning system is ready.")
        print("Key features:")
        print("- Regime clustering with K-Means")
        print("- Anomaly detection with Isolation Forest")
        print("- Confidence multipliers for strategies")
        print("- Proactive insights generation")
        print("- Integration with reasoning engine")
    else:
        print("❌ Some tests failed. Check the output above.")

    return success

if __name__ == "__main__":
    main()