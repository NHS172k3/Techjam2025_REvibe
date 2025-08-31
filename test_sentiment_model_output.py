import pandas as pd
import numpy as np
from analyze_sentiment_quality import process_video_comments, test_sentiment_model
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_sentiment_test():
    """
    Comprehensive test of sentiment model with various comment types
    """
    print("🧪 COMPREHENSIVE SENTIMENT MODEL TEST")
    print("="*50)
    
    # Test cases with expected sentiment ranges
    test_cases = [
        # Very Positive (Expected: 0.8-1.0)
        {
            'name': 'Very Positive Comments',
            'comments': [
                "This is absolutely amazing! Best video ever!",
                "Incredible content! I love everything about this!",
                "Outstanding work! This made my day so much better!",
                "Perfect! Exactly what I was looking for!",
                "Brilliant! This is pure genius!"
            ],
            'expected_range': (0.8, 1.0)
        },
        
        # Positive (Expected: 0.6-0.8)
        {
            'name': 'Positive Comments',
            'comments': [
                "Really good video, enjoyed watching it",
                "Nice work, keep it up!",
                "I liked this, thanks for sharing",
                "Good content, well done",
                "Pretty cool, I enjoyed it"
            ],
            'expected_range': (0.6, 0.8)
        },
        
        # Neutral (Expected: 0.4-0.6)
        {
            'name': 'Neutral Comments',
            'comments': [
                "It's okay, nothing special",
                "This was fine I guess",
                "Average content, not bad not great",
                "Watched it, it was alright",
                "Meh, could be better"
            ],
            'expected_range': (0.4, 0.6)
        },
        
        # Negative (Expected: 0.2-0.4)
        {
            'name': 'Negative Comments',
            'comments': [
                "Not really my thing, didn't like it",
                "This was boring and uninteresting",
                "Disappointing content, expected better",
                "Not good, waste of time",
                "Poor quality, needs improvement"
            ],
            'expected_range': (0.2, 0.4)
        },
        
        # Very Negative (Expected: 0.0-0.2)
        {
            'name': 'Very Negative Comments',
            'comments': [
                "Absolutely terrible! Worst video ever!",
                "Completely awful, I hate this!",
                "Disgusting content, this is horrible!",
                "Terrible quality, this is garbage!",
                "Utterly disappointing, complete waste!"
            ],
            'expected_range': (0.0, 0.2)
        },
        
        # Mixed Comments (Expected: ~0.5)
        {
            'name': 'Mixed Comments',
            'comments': [
                "Great video! | This was terrible | Pretty good content | Not my favorite | Amazing work!"
            ],
            'expected_range': (0.4, 0.6)
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 Testing: {test_case['name']}")
        print(f"Expected range: {test_case['expected_range'][0]:.1f} - {test_case['expected_range'][1]:.1f}")
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'video_id': [f'test_{test_case["name"].lower().replace(" ", "_")}_{j}' for j in range(len(test_case['comments']))],
            'top_comments': test_case['comments']
        })
        
        # Process with sentiment model
        result_df = process_video_comments(test_df)
        
        # Analyze results
        sentiment_scores = result_df['sentiment_score'].values
        avg_sentiment = np.mean(sentiment_scores)
        min_sentiment = np.min(sentiment_scores)
        max_sentiment = np.max(sentiment_scores)
        
        print(f"   Scores: {sentiment_scores}")
        print(f"   Average: {avg_sentiment:.3f}")
        print(f"   Range: {min_sentiment:.3f} - {max_sentiment:.3f}")
        
        # Check if in expected range
        in_range = (test_case['expected_range'][0] <= avg_sentiment <= test_case['expected_range'][1])
        status = "✅ PASS" if in_range else "❌ FAIL"
        print(f"   Status: {status}")
        
        # Store results
        results.append({
            'test_case': test_case['name'],
            'expected_min': test_case['expected_range'][0],
            'expected_max': test_case['expected_range'][1],
            'actual_avg': avg_sentiment,
            'actual_min': min_sentiment,
            'actual_max': max_sentiment,
            'in_range': in_range,
            'scores': sentiment_scores
        })
    
    return results

def plot_sentiment_results(results):
    """Create visualization of sentiment test results"""
    
    print("\n📈 Creating sentiment test visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Expected vs Actual sentiment ranges
    test_names = [r['test_case'] for r in results]
    expected_mins = [r['expected_min'] for r in results]
    expected_maxs = [r['expected_max'] for r in results]
    actual_avgs = [r['actual_avg'] for r in results]
    
    x_pos = np.arange(len(test_names))
    
    # Plot expected ranges as bars
    ax1.bar(x_pos, [max_val - min_val for min_val, max_val in zip(expected_mins, expected_maxs)], 
            bottom=expected_mins, alpha=0.3, label='Expected Range', color='blue')
    
    # Plot actual averages as dots
    colors = ['green' if r['in_range'] else 'red' for r in results]
    ax1.scatter(x_pos, actual_avgs, color=colors, s=100, label='Actual Average', zorder=3)
    
    ax1.set_xlabel('Test Cases')
    ax1.set_ylabel('Sentiment Score (0-1)')
    ax1.set_title('Sentiment Model Test Results')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.replace(' Comments', '') for name in test_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Distribution of all scores
    all_scores = []
    labels = []
    for result in results:
        for score in result['scores']:
            all_scores.append(score)
            labels.append(result['test_case'].replace(' Comments', ''))
    
    # Create box plot
    data_for_box = [result['scores'] for result in results]
    box_labels = [r['test_case'].replace(' Comments', '') for r in results]
    
    ax2.boxplot(data_for_box, labels=box_labels)
    ax2.set_xlabel('Test Cases')
    ax2.set_ylabel('Sentiment Score (0-1)')
    ax2.set_title('Distribution of Sentiment Scores')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sentiment_model_test_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'sentiment_model_test_results.png'")
    
    return fig

def test_model_consistency():
    """Test model consistency with repeated runs"""
    print("\n🔄 Testing Model Consistency...")
    
    test_comment = "This video is really good! I enjoyed watching it."
    test_df = pd.DataFrame({
        'video_id': ['consistency_test'],
        'top_comments': [test_comment]
    })
    
    scores = []
    for i in range(5):
        result = process_video_comments(test_df)
        scores.append(result['sentiment_score'].iloc[0])
        print(f"   Run {i+1}: {scores[-1]:.4f}")
    
    variance = np.var(scores)
    print(f"   Variance: {variance:.6f}")
    
    if variance < 0.001:
        print("   ✅ Model is consistent (low variance)")
    else:
        print("   ⚠️ Model shows some inconsistency")
    
    return scores

def generate_summary_report(results):
    """Generate a summary report of the test results"""
    
    print("\n📋 SENTIMENT MODEL TEST SUMMARY")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['in_range'])
    
    print(f"Total test cases: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n📊 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result['in_range'] else "❌"
        print(f"{status} {result['test_case']}: {result['actual_avg']:.3f} (expected: {result['expected_min']:.1f}-{result['expected_max']:.1f})")
    
    # Performance analysis
    print(f"\n🎯 MODEL PERFORMANCE:")
    
    # Check if model can distinguish between very positive and very negative
    very_pos = next(r for r in results if 'Very Positive' in r['test_case'])
    very_neg = next(r for r in results if 'Very Negative' in r['test_case'])
    
    separation = very_pos['actual_avg'] - very_neg['actual_avg']
    print(f"Positive-Negative separation: {separation:.3f}")
    
    if separation > 0.6:
        print("✅ Good separation between positive and negative sentiment")
    elif separation > 0.4:
        print("⚠️ Moderate separation between positive and negative sentiment")
    else:
        print("❌ Poor separation between positive and negative sentiment")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests/total_tests*100,
        'separation': separation
    }

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Sentiment Model Testing...")
    
    # Run basic test first
    print("\n1️⃣ Basic Model Test:")
    test_sentiment_model()
    
    # Run comprehensive test
    print("\n2️⃣ Comprehensive Test:")
    results = comprehensive_sentiment_test()
    
    # Test consistency
    print("\n3️⃣ Consistency Test:")
    consistency_scores = test_model_consistency()
    
    # Generate visualization
    print("\n4️⃣ Generating Visualization:")
    try:
        plot_sentiment_results(results)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    # Generate summary
    print("\n5️⃣ Summary Report:")
    summary = generate_summary_report(results)
    
    print(f"\n🏁 Testing Complete!")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Sentiment Separation: {summary['separation']:.3f}")
    
    if summary['success_rate'] >= 80 and summary['separation'] >= 0.4:
        print("🎉 Sentiment model is performing well!")
    elif summary['success_rate'] >= 60:
        print("⚠️ Sentiment model needs some improvement")
    else:
        print("❌ Sentiment model needs significant improvement")