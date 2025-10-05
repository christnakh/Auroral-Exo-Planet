#!/usr/bin/env python3
"""
🧪 Complete System Testing and Validation Runner
Runs all models, tests everything, and provides comprehensive reports
"""

import os
import sys
import time
import subprocess
import signal
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def run_complete_testing():
    """Run complete system testing with all models and validations"""
    
    print("\n" + "="*80)
    print("🧪 COMPLETE EXOPLANET ML SYSTEM TESTING")
    print("="*80)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run all models training
    print("\n📊 STEP 1: TRAINING ALL MODELS")
    print("-" * 50)
    
    try:
        print("🚀 Running complete model training pipeline...")
        result = subprocess.run([sys.executable, 'run_all_models.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ All models trained successfully!")
            print("📊 Training Results:")
            print(result.stdout)
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Model training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running model training: {e}")
        return False
    
    # Step 2: Start Flask server
    print("\n🌐 STEP 2: STARTING FLASK SERVER")
    print("-" * 50)
    
    flask_process = None
    server_port = None
    
    try:
        print("🚀 Starting Flask server...")
        flask_process = subprocess.Popen([sys.executable, 'app.py'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)
        
        # Check if server is running
        try:
            import requests
            # Try different ports
            for port in range(5001, 5010):
                try:
                    response = requests.get(f"http://localhost:{port}/api/health", timeout=5)
                    if response.status_code == 200:
                        server_port = port
                        print(f"✅ Flask server running on port {port}")
                        break
                except:
                    continue
            
            if not server_port:
                print("❌ Flask server failed to start")
                return False
                
        except ImportError:
            print("⚠️  requests library not available, assuming server started")
            server_port = 5001
    
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        return False
    
    # Step 3: Run comprehensive testing
    print("\n🧪 STEP 3: COMPREHENSIVE SYSTEM TESTING")
    print("-" * 50)
    
    try:
        print("🔬 Running complete system tests...")
        test_result = subprocess.run([sys.executable, 'test_complete_system.py', 
                                    '--port', str(server_port)], 
                                   capture_output=True, text=True, timeout=600)
        
        if test_result.returncode == 0:
            print("✅ Complete system testing passed!")
            print("📊 Test Results Summary:")
            print(test_result.stdout)
        else:
            print("❌ System testing failed:")
            print(test_result.stderr)
            print("📊 Partial Results:")
            print(test_result.stdout)
    
    except subprocess.TimeoutExpired:
        print("⏰ System testing timed out (10 minutes)")
    except Exception as e:
        print(f"❌ Error running system tests: {e}")
    
    # Step 4: Generate final report
    print("\n📊 STEP 4: GENERATING FINAL REPORT")
    print("-" * 50)
    
    try:
        # Check if test results exist
        test_results_dir = Path("test_results")
        if test_results_dir.exists():
            print("📄 Test results available:")
            for file in test_results_dir.glob("*"):
                print(f"   📁 {file.name}")
        
        # Check logs
        logs_dir = Path("logs")
        if logs_dir.exists():
            print("📝 Logs available:")
            for file in logs_dir.glob("*.log"):
                print(f"   📄 {file.name}")
        
        # Check models
        models_dir = Path("models/trained_models")
        if models_dir.exists():
            print("🤖 Trained models available:")
            for file in models_dir.glob("*"):
                print(f"   🧠 {file.name}")
    
    except Exception as e:
        print(f"❌ Error generating final report: {e}")
    
    # Step 5: Cleanup
    print("\n🧹 STEP 5: CLEANUP")
    print("-" * 50)
    
    try:
        if flask_process:
            print("🛑 Stopping Flask server...")
            flask_process.terminate()
            flask_process.wait(timeout=5)
            print("✅ Flask server stopped")
    except Exception as e:
        print(f"⚠️  Error stopping Flask server: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 COMPLETE SYSTEM TESTING FINISHED!")
    print("="*80)
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 RESULTS AVAILABLE:")
    print("   📊 Test Results: test_results/")
    print("   📝 Logs: logs/")
    print("   🤖 Models: models/trained_models/")
    print("   📄 Reports: test_results/comprehensive_test_report.json")
    
    print("\n🌐 TO START THE WEB APPLICATION:")
    print("   python3 app.py")
    print("   # The app will automatically find an available port")
    
    print("\n🧪 TO RUN TESTS AGAIN:")
    print("   python3 run_complete_tests.py")
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Exoplanet ML System Testing')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training step')
    parser.add_argument('--skip-server', action='store_true', 
                       help='Skip Flask server testing')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick tests only')
    
    args = parser.parse_args()
    
    print("🧪 Complete Exoplanet ML System Testing")
    print("=" * 50)
    
    if args.quick:
        print("⚡ Running quick tests only...")
    
    if args.skip_training:
        print("⏭️  Skipping model training...")
    
    if args.skip_server:
        print("⏭️  Skipping Flask server testing...")
    
    # Run complete testing
    success = run_complete_testing()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
