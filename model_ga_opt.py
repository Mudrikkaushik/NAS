import torch, random, os, json
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy

from model_cnn import CNN


class CNNSearchSpace:
    def __init__(self):
        self.conv_layers = [1, 2, 3, 4]
        self.filters = [16, 32, 64, 128]
        self.kernel_sizes = [3, 5, 7]
        self.pool_types = ['max', 'avg']
        self.activations = ['relu', 'leaky_relu']
        self.fc_units = [64, 128, 256, 512]


class Architecture:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        self.fitness = 0
        self.accuracy = 0
        self.best_epoch = 0
    
    def random_genes(self):
        space = CNNSearchSpace()
        num_conv = random.choice(space.conv_layers)
        
        genes = {
            'num_conv': num_conv,
            'conv_configs': [],
            'pool_type': random.choice(space.pool_types),
            'activation': random.choice(space.activations),
            'fc_units': random.choice(space.fc_units)
        }
        
        for _ in range(num_conv):
            genes['conv_configs'].append({
                'filters': random.choice(space.filters),
                'kernel_size': random.choice(space.kernel_sizes)
            })
        
        return genes
    
    def __repr__(self):
        return f"Arch(conv={self.genes['num_conv']}, acc={self.accuracy:.4f})"


class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_architecture = None
        self.search_space = CNNSearchSpace()
    
    def initialize_population(self):
        self.population = [Architecture() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, architecture, train_loader, val_loader, device, epochs=100):
        try:
            model = CNN(architecture.genes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=0.001)
            
            # training loop
            best_acc = 0
            patience = 10
            step = 1
            best_epoch = 1

            for epoch in range(1, epochs+1):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
                # eval
                model.eval()
                correct = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
            
                accuracy = correct / len(val_loader.dataset)

                if accuracy > best_acc:
                    step = 0
                    best_acc = accuracy
                    best_epoch = epoch
                else:
                    step += 1

                if step >= patience:
                    break

            # -------------------------------
            # Parameter counting: Conv vs FC
            # -------------------------------
            conv_params = 0
            fc_params = 0

            for name, param in model.named_parameters():
                n = param.numel()
                lname = name.lower()

                if "conv" in lname or "bn" in lname:
                    conv_params += n
                elif "fc" in lname or "linear" in lname:
                    fc_params += n
                else:
                    if len(param.shape) >= 4:
                        conv_params += n
                    else:
                        fc_params += n

            conv_m = conv_params / 1e6
            fc_m = fc_params / 1e6

            alpha = 1e-3
            beta = 5e-4

            penalty = alpha * conv_m + beta * fc_m

            architecture.accuracy = best_acc
            architecture.best_epoch = best_epoch
            architecture.fitness = best_acc - penalty

            # logging required
            print(f"Conv Params: {conv_m:.6f}M | FC Params: {fc_m:.6f}M")
            print(f"Penalty = {alpha}*{conv_m:.6f} + {beta}*{fc_m:.6f} = {penalty:.6f}")
            print(f"Final Fitness = {architecture.fitness:.6f}")

            del model
            torch.cuda.empty_cache()

            return architecture.fitness

        except Exception as e:
            print(f"Error evaluating architecture: {e}", flush=True)
            architecture.fitness = 0
            architecture.accuracy = 0
            return 0
    

    def selection(self):
        fitness_vals = [arch.fitness for arch in self.population]

        eps = 1e-12
        fitness_vals = [0.0 if f is None or (isinstance(f, float) and f != f) else f for f in fitness_vals]
        min_f = min(fitness_vals)

        if min_f <= 0:
            shift = abs(min_f) + 1e-6
            fitness_shifted = [f + shift for f in fitness_vals]
        else:
            fitness_shifted = fitness_vals

        total_f = sum(fitness_shifted) + eps
        probs = [f / total_f for f in fitness_shifted]

        cumulative = []
        csum = 0
        for p in probs:
            csum += p
            cumulative.append(csum)

        print("\n[SELECTION LOG] Fitness → Probability:")
        for idx, (f, p) in enumerate(zip(fitness_vals, probs)):
            print(f"  {idx}: fitness={f:.6f} → prob={p:.6f}")

        selected = []
        for _ in range(self.population_size):
            r = random.random()
            for idx, c in enumerate(cumulative):
                if r <= c:
                    selected.append(self.population[idx])
                    break

        return selected
    

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1_genes = deepcopy(parent1.genes)
        child2_genes = deepcopy(parent2.genes)
        
        if random.random() < 0.5:
            child1_genes['num_conv'], child2_genes['num_conv'] = child2_genes['num_conv'], child1_genes['num_conv']
        
        if random.random() < 0.5:
            child1_genes['pool_type'], child2_genes['pool_type'] = child2_genes['pool_type'], child1_genes['pool_type']
            child1_genes['activation'], child2_genes['activation'] = child2_genes['activation'], child1_genes['activation']
        
        min_len = min(child1_genes['num_conv'], len(child1_genes['conv_configs']))
        child1_genes['conv_configs'] = child1_genes['conv_configs'][:min_len]
        while len(child1_genes['conv_configs']) < child1_genes['num_conv']:
            child1_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        min_len = min(child2_genes['num_conv'], len(child2_genes['conv_configs']))
        child2_genes['conv_configs'] = child2_genes['conv_configs'][:min_len]
        while len(child2_genes['conv_configs']) < child2_genes['num_conv']:
            child2_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        return Architecture(child1_genes), Architecture(child2_genes)
    

    def mutation(self, architecture):
        if random.random() > self.mutation_rate:
            return architecture
        
        genes = deepcopy(architecture.genes)
        mutation_type = random.choice(['conv_param', 'num_layers', 'pool_activation', 'fc_units'])
        
        if mutation_type == 'conv_param' and genes['conv_configs']:
            idx = random.randint(0, len(genes['conv_configs']) - 1)
            genes['conv_configs'][idx]['filters'] = random.choice(self.search_space.filters)
            genes['conv_configs'][idx]['kernel_size'] = random.choice(self.search_space.kernel_sizes)
        
        elif mutation_type == 'num_layers':
            genes['num_conv'] = random.choice(self.search_space.conv_layers)
            if genes['num_conv'] > len(genes['conv_configs']):
                for _ in range(genes['num_conv'] - len(genes['conv_configs'])):
                    genes['conv_configs'].append({
                        'filters': random.choice(self.search_space.filters),
                        'kernel_size': random.choice(self.search_space.kernel_sizes)
                    })
            else:
                genes['conv_configs'] = genes['conv_configs'][:genes['num_conv']]
        
        elif mutation_type == 'pool_activation':
            genes['pool_type'] = random.choice(self.search_space.pool_types)
            genes['activation'] = random.choice(self.search_space.activations)
        
        elif mutation_type == 'fc_units':
            genes['fc_units'] = random.choice(self.search_space.fc_units)
        
        return Architecture(genes)


    def evolve(self, train_loader, val_loader, device, run=1):
        parent = os.path.abspath('')
        self.initialize_population()

        print(f"Starting Population ({self.population_size}):\n{self.population}\n")

        for generation in range(self.generations):

            print("\n" + "="*60)
            print(f"Generation {generation+1}/{self.generations}")
            print("="*60)

            # fitness
            for i, arch in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{self.population_size}...", end=' ')
                fitness = self.evaluate_fitness(arch, train_loader, val_loader, device)
                print(f"Fitness: {fitness:.4f}, Accuracy: {arch.accuracy:.4f}")

            # sorting
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            if self.best_architecture is None or self.population[0].fitness > self.best_architecture.fitness:
                self.best_architecture = deepcopy(self.population[0])

            print(f"Best of Generation: {self.population[0]}")
            print(f"Best Overall: {self.best_architecture}")

            # selection
            print("\nPerforming roulette-wheel selection ...")
            selected = self.selection()

            # next population
            next_generation = []
            print("Elitism: keeping top 2...")
            next_generation.extend([deepcopy(self.population[0]), deepcopy(self.population[1])])

            while len(next_generation) < self.population_size:
                p1 = random.choice(selected)
                p2 = random.choice(selected)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)

                next_generation.append(c1)
                if len(next_generation) < self.population_size:
                    next_generation.append(c2)

            self.population = next_generation
            print("Next Generation:", self.population)

            # save output
            out = os.path.join(parent, "outputs", f"run_{run}")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"generation_{generation}.jsonl"), "w") as f:
                for obj in self.population:
                    f.write(json.dumps(obj.genes) + "\n")

        return self.best_architecture
