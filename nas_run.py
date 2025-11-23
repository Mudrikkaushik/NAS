import torch, sys, os, pickle
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from model_ga import GeneticAlgorithm
from model_cnn import CNN

# if __name__ == "__main__":

parent = os.path.abspath('')
if not os.path.exists(os.path.join(parent, 'outputs')):
    os.mkdir(os.path.join(parent, 'outputs'))
all_runs = [i for i in os.listdir(os.path.join(parent, 'outputs')) if 'run_' in i]
run_num = len(all_runs) + 1
run_dir = os.path.join(parent, 'outputs', f'run_{run_num}')
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable cuDNN benchmark for faster training with fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

sys.stdout = open(os.path.join(run_dir, f'nas_run.log'), 'w')

print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"cuDNN benchmark enabled: {torch.backends.cudnn.benchmark}", flush=True)

# Load CIFAR-10 dataset (reduced for faster NAS)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use only 5000 samples for quick NAS
train_subset = Subset(trainset, range(5000))
val_subset = Subset(valset, range(1000))

# Optimize DataLoader with multiple workers and pinned memory for GPU
num_workers = 4 if torch.cuda.is_available() else 0
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)

# Run NAS with GA
ga = GeneticAlgorithm(
    population_size=10,  # Small population for demonstration
    generations=5,       # Few generations for quick results
    mutation_rate=0.3,
    crossover_rate=0.7
)

best_arch = ga.evolve(train_loader, val_loader, device, run=run_num)

print(f"\n{'='*60}", flush=True)
print("FINAL BEST ARCHITECTURE", flush=True)
print(f"{'='*60}", flush=True)
print(f"Genes: {best_arch.genes}", flush=True)
print(f"Accuracy: {best_arch.accuracy:.4f}", flush=True)
print(f"Fitness: {best_arch.fitness:.4f}", flush=True)

# Build and test final model
final_model = CNN(best_arch.genes).to(device)
print(f"\nTotal parameters: {sum(p.numel() for p in final_model.parameters()):,}", flush=True)
print(f"\nModel architecture:\n{final_model}", flush=True)

with open(os.path.join(run_dir, f"best_arch.pkl"), 'wb') as f:
    pickle.dump(best_arch, f)

sys.stdout = sys.__stdout__