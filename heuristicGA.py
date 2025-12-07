import numpy as np
import random
import copy
import time
from route import Route

class GeneticAlgorithm:
    def __init__(self, user_param, pop_size=100, generations=100, crossover_prob=0.8, mutation_prob=0.1):
        """
        初始化遗传算法
        :param user_param: ParamsVRP 实例 (包含所有成本参数和计算方法)
        :param pop_size: 种群大小
        :param generations: 迭代代数
        """
        self.params = user_param
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []  # 存储个体: {'genome': [], 'cost': inf, 'routes': []}

    def run(self):
        """
        运行遗传算法的主函数
        :return: (best_routes, best_cost, run_time)
        """
        start_time = time.time()
        
        # 1. 初始化种群
        self.initialize_population()
        
        best_solution = None
        no_improvement_count = 0
        
        for gen in range(self.generations):
            # 2. 评估适应度 (解码 + 计算成本)
            # 使用 Split Procedure 将 genome 切割为 routes，并计算 LCCC-VRP-STW 成本
            for ind in self.population:
                if ind['cost'] is None:
                    self.split_procedure(ind)
            
            # 3. 排序并记录最优
            self.population.sort(key=lambda x: x['cost'])
            current_best = self.population[0]
            
            if best_solution is None or current_best['cost'] < best_solution['cost']:
                best_solution = copy.deepcopy(current_best)
                no_improvement_count = 0
                # print(f"Gen {gen}: New Best Cost = {best_solution['cost']:.2f}")
            else:
                no_improvement_count += 1
            
            # 早停机制 (可选)
            if no_improvement_count > 50:
                # print(f"Gen {gen}: Early stopping due to no improvement.")
                break

            # 4. 生成新一代 (精英保留 + 锦标赛选择 + 交叉 + 变异)
            new_pop = [copy.deepcopy(self.population[0])] # 保留最好的1个
            
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                # 交叉
                if random.random() < self.crossover_prob:
                    child_genome = self.crossover_ox(p1['genome'], p2['genome'])
                else:
                    child_genome = p1['genome'][:] # 直接复制
                
                # 变异
                if random.random() < self.mutation_prob:
                    child_genome = self.mutation(child_genome)
                
                new_pop.append({'genome': child_genome, 'cost': None, 'routes': []})
            
            self.population = new_pop

        end_time = time.time()
        return best_solution['routes'], best_solution['cost'], end_time - start_time

    def initialize_population(self):
        """生成随机的大周游序列 (Giant Tour)"""
        # 客户编号从 1 到 nbclients-2 (因为 nbclients-1 是 End Depot)
        base_genome = list(range(1, self.params.nbclients - 1))
        
        for _ in range(self.pop_size):
            random.shuffle(base_genome)
            self.population.append({'genome': base_genome[:], 'cost': None, 'routes': []})

    def split_procedure(self, individual):
        """
        Prins Split Procedure (核心解码函数)
        将 Giant Tour (染色体) 切割为最优的车辆路径组合。
        这里使用 Dijkstra/Bellman-Ford 思想在辅助图上找最短路。
        """
        genome = individual['genome']
        n = len(genome)
        
        # V[i] 表示前 i 个客户已经被服务时的最小累积成本
        V = [float('inf')] * (n + 1)
        # P[i] 记录前驱节点，用于回溯路径
        P = [-1] * (n + 1)
        V[0] = 0
        
        # 遍历所有可能的切割点
        for i in range(n):
            if V[i] == float('inf'):
                continue
                
            load = 0
            # 尝试构建路径: Depot -> genome[i] ... -> genome[j] -> Depot
            for j in range(i, n):
                client = genome[j]
                
                # 1. 容量约束 (硬约束)
                load += self.params.d[client]
                if load > self.params.capacity:
                    break # 超载，当前车辆无法再装更多，停止向后探索
                
                # 2. 构建临时路径片段 [0, c_i, ..., c_j, 0_end]
                # 注意：paramsVRP 中的 calculate_actual_cost 需要完整的节点列表
                sub_route_clients = genome[i : j+1]
                path_segment = [0] + sub_route_clients + [self.params.nbclients - 1]
                
                # 3. 计算复杂成本 (包含：行驶、碳税、制冷、新鲜度、软时间窗惩罚)
                # 直接调用 paramsVRP 中定义好的函数，保证与 B&P 逻辑一致
                route_cost = self.params.calculate_actual_cost(path_segment)
                
                # *重要*: calculate_actual_cost 内部可以检查可行性吗？
                # 如果你的软时间窗允许无限迟到(只是罚款)，那总是可行的。
                # 如果有最大迟到限制 (max_lateness)，我们需要额外检查。
                # 这里假设 calculate_actual_cost 如果返回非常大的值代表不可行，或者我们手动检查
                
                # 4. 手动检查最大迟到 (以防 calculate_actual_cost 只是算钱)
                # 简单估算一下最后一个点的到达时间是否极其离谱 (可选优化)
                
                # 5. 更新最短路 (松弛操作)
                if V[i] + route_cost < V[j+1]:
                    V[j+1] = V[i] + route_cost
                    P[j+1] = i

        # 回溯生成 Route 对象
        routes = []
        curr = n
        
        if V[n] == float('inf'):
            individual['cost'] = float('inf')
            individual['routes'] = []
            return

        while curr > 0:
            prev = P[curr]
            sub_route_clients = genome[prev:curr]
            full_path = [0] + sub_route_clients + [self.params.nbclients - 1]
            
            # 计算该路径的具体 cost 和 Q
            cost = self.params.calculate_actual_cost(full_path)
            q_val = sum(self.params.d[c] for c in sub_route_clients)
            
            routes.append(Route(path=full_path, cost=cost, Q=q_val))
            curr = prev
            
        individual['routes'] = routes
        individual['cost'] = V[n]

    def crossover_ox(self, p1, p2):
        """顺序交叉 (Order Crossover)"""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        
        # 继承 P1 的片段
        child[a:b] = p1[a:b]
        
        # 按顺序填充 P2 的剩余基因
        pointer = b
        for gene in p2:
            if gene not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        return child

    def mutation(self, genome):
        """交换变异"""
        idx1, idx2 = random.sample(range(len(genome)), 2)
        genome[idx1], genome[idx2] = genome[idx2], genome[idx1]
        return genome

    def tournament_selection(self):
        """锦标赛选择"""
        k = 3
        candidates = random.sample(self.population, k)
        return min(candidates, key=lambda x: x['cost'])