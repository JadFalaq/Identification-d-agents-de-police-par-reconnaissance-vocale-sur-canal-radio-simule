import torch
import time
import sys


print(" TEST CONFIGURATION GPU - GTX 1650")


# Test 1: PyTorch
print("\n PyTorch:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   Version CUDA: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute: {torch.cuda.get_device_capability(0)}")
    print(f"   VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test 2: Calcul GPU
    print("\n2 Test performance:")
    
    # GPU
    x = torch.randn(2000, 2000, device='cuda')
    y = torch.randn(2000, 2000, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   GPU: {gpu_time*1000:.2f}ms")
    
    # CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start = time.time()
    z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"   CPU: {cpu_time*1000:.2f}ms")
    print(f"   Accélération: {cpu_time/gpu_time:.1f}x")
    
    # Test 3: Mémoire
    print("\n3️ Mémoire GPU:")
    print(f"   Allouée: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    print(f"   Réservée: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")
    print(f"   Disponible: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB")
    
    # Test 4: Batch processing (simulation)
    print("\n4️ Test batch processing:")
    batch_sizes = [8, 16, 32, 64]
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, 3, 224, 224, device='cuda')
            y = torch.randn(bs, 1000, device='cuda')
            mem = torch.cuda.memory_allocated(0) / 1e6
            print(f"   Batch {bs:2d}: {mem:.0f}MB ")
            del x, y
        except RuntimeError as e:
            print(f"   Batch {bs:2d}: Out of Memory ")
            break
    
    torch.cuda.empty_cache()
    print("\n GPU fonctionnel et prêt !")
    
else:
    print("\n CUDA non disponible")
    print("Vérifier l'installation PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


