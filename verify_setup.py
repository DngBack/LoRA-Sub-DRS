#!/usr/bin/env python3
"""
HDRS Environment Verification Script

This script verifies that all required packages are installed correctly
and that the environment is ready for HDRS experiments.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def check_basic_imports():
    """Check basic Python packages"""
    print("=== HDRS Environment Verification ===")
    print(f"Python version: {sys.version}")
    
    # Check core packages
    try:
        import torch
        import torchvision
        import timm
        import geoopt
        import numpy as np
        import scipy
        import sklearn
        import yaml
        import tqdm
        
        print("✅ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_versions():
    """Check package versions"""
    import torch
    import torchvision
    import timm
    import geoopt
    import numpy as np
    import scipy
    import sklearn
    
    print(f"\n=== Package Versions ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"Timm: {timm.__version__}")
    print(f"Geoopt: {geoopt.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    
    # Check for problematic NumPy version
    if np.__version__.startswith('2.'):
        print("⚠️  WARNING: NumPy 2.x detected. This may cause compatibility issues.")
        print("   Recommended: pip install numpy==1.24.3 --force-reinstall")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability and configuration"""
    import torch
    
    print(f"\n=== CUDA Configuration ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        return True
    else:
        print("❌ CUDA not available. GPU acceleration disabled.")
        print("   Make sure you have a compatible NVIDIA GPU and drivers.")
        return False

def test_functionality():
    """Test critical functionality"""
    print(f"\n=== Functionality Tests ===")
    
    try:
        # Test geoopt manifolds
        from geoopt import Stiefel, Sphere
        import torch
        
        # Test Stiefel manifold
        stiefel = Stiefel()
        x = torch.randn(10, 5)
        x_proj = stiefel.projx(x)
        print("✅ Stiefel manifold operations working")
        
        # Test Sphere manifold  
        sphere = Sphere()
        y = torch.randn(10)
        y_proj = sphere.projx(y)
        print("✅ Sphere manifold operations working")
        
    except Exception as e:
        print(f"❌ Manifold operations failed: {e}")
        return False
    
    try:
        # Test vision transformer creation
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        print("✅ Vision Transformer creation working")
        
    except Exception as e:
        print(f"❌ Vision Transformer creation failed: {e}")
        return False
    
    # Test CUDA operations if available
    if torch.cuda.is_available():
        try:
            x = torch.randn(2, 3, 224, 224).cuda()
            model = model.cuda()
            with torch.no_grad():
                y = model(x)
            print("✅ CUDA tensor operations working")
            
            # Test memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb < 8:
                print(f"⚠️  WARNING: Only {memory_gb:.1f}GB GPU memory available.")
                print("   Recommended: At least 8GB for basic experiments, 24GB for full scale.")
                
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
            return False
    
    return True

def check_dataset():
    """Check if CIFAR-100 dataset can be loaded"""
    print(f"\n=== Dataset Test ===")
    
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        
        # Test CIFAR-100 loading
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
        print(f"✅ CIFAR-100 dataset accessible ({len(dataset)} samples)")
        
    except Exception as e:
        print(f"ℹ️  CIFAR-100 not downloaded yet (normal for first run)")
        print("   Will be downloaded automatically when running experiments")
    
    return True

def test_hdrs_imports():
    """Test HDRS-specific imports"""
    print(f"\n=== HDRS Code Test ===")
    
    try:
        # Test if we can import the main modules
        sys.path.append('.')
        
        # These imports will fail if there are syntax errors
        from methods import hdrs
        from models import riemannian_lora
        print("✅ HDRS modules can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ HDRS module import failed: {e}")
        print("   This might indicate syntax errors in the code")
        return False

def main():
    """Run all verification checks"""
    print("🚀 Starting HDRS environment verification...\n")
    
    checks = [
        ("Basic Imports", check_basic_imports),
        ("Package Versions", check_versions), 
        ("CUDA Support", check_cuda),
        ("Core Functionality", test_functionality),
        ("Dataset Access", check_dataset),
        ("HDRS Code", test_hdrs_imports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {name}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All checks passed! Environment is ready for HDRS experiments.")
        print("\nYou can now run:")
        print("  python main.py --config configs/hdrs_cifar100_small.json")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("   See SETUP_GUIDE.md for troubleshooting instructions.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
