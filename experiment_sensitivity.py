import numpy as np
import time
import os
from branchBound import BranchAndBound
from heuristicGA import GeneticAlgorithm # [新增]
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis

# ==========================================
# [配置区] 选择你要使用的算法
# 建议灵敏度分析使用 "GA"，因为需要运行多次
# ==========================================
ALGORITHM_TYPE = "GA" 
# ALGORITHM_TYPE = "BP"

def run_single_experiment(datasetPath, c_tax=None, max_lateness=None, p_fresh=None, theta=None, label="", save_fig_path=None):
    # 1. 初始化
    user_param = ParamsVRP()
    # 灵敏度分析通常使用 N=50 或 N=100 的固定数据集
    user_param.init_params(datasetPath) 
    # 如果你想跑得快一点，可以限制客户数，例如：
    # user_param.init_params(datasetPath, max_customers=50)
    
    # 2. 覆盖参数 (Sensitivity Override)
    if c_tax is not None:
        user_param.C_tax = c_tax
    if max_lateness is not None:
        user_param.max_lateness = max_lateness
    if p_fresh is not None:
        user_param.P_fresh = p_fresh
    if theta is not None:
        user_param.theta = theta
        
    # [重要] 重新计算静态成本矩阵
    # 因为 C_tax 等参数改变了，edge 的权重也变了，必须刷新
    unit_dist_cost = user_param.rho_avg * (user_param.P_fuel + user_param.eta_CO2 * user_param.C_tax) + user_param.beta_ref
    for i in range(user_param.nbclients):
        for j in range(user_param.nbclients):
            user_param.static_cost[i][j] = user_param.dist[i][j] * unit_dist_cost
            # 重置 cost 矩阵
            user_param.cost[i][j] = user_param.static_cost[i][j]

    best_routes = []
    start_time = time.time()

    # ==========================================
    # 分支 1: GA
    # ==========================================
    if ALGORITHM_TYPE == "GA":
        # 灵敏度分析不需要跑太多代，80-100代足够看趋势
        ga = GeneticAlgorithm(user_param, pop_size=80, generations=80)
        best_routes, _, _ = ga.run()

    # ==========================================
    # 分支 2: B&P
    # ==========================================
    elif ALGORITHM_TYPE == "BP":
        bp = BranchAndBound()
        init_routes = []
        for i in range(user_param.nbclients - 2):
            path = [0, i + 1, user_param.nbclients - 1]
            route_cost = user_param.calculate_actual_cost(path)
            route = Route(path=path, cost=route_cost, Q=1.0)
            init_routes.append(route)
        
        bp.bb_node(user_param, init_routes, None, best_routes, 0)
    
    else:
        raise ValueError("Unknown ALGORITHM_TYPE")

    end_time = time.time()
    sol_time = end_time - start_time
    
    # ==========================================
    # 3. 结果统计 (通用)
    # ==========================================
    total_cost = 0
    total_dist = 0
    total_emission = 0 # kg
    total_fresh_loss = 0 # CNY
    
    for route in best_routes:
        total_cost += route.get_cost()
        path = route.get_path()
        current_time = 0.0
        
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            
            d_ij = user_param.dist[i][j]
            # Fix infinity distance issue for stats
            if d_ij >= user_param.verybig / 100:
                d_ij = user_param.dist_base[i][j]
                
            total_dist += d_ij
            total_emission += d_ij * user_param.rho_avg * user_param.eta_CO2
            
            # Time update
            if k == 0: departure = 0
            else: departure = max(current_time, user_param.a[i]) + user_param.s[i]
            
            current_time = departure + user_param.ttime[i][j]
            
            if j != 0 and j != user_param.nbclients - 1:
                freshness_cost = user_param.P_fresh * user_param.d[j] * user_param.theta * current_time
                total_fresh_loss += freshness_cost

    # Calculate Rates
    avg_dist = total_dist / len(best_routes) if len(best_routes) > 0 else 0
    total_goods_value = sum(user_param.d[1:-1]) * user_param.P_fresh
    loss_rate = (total_fresh_loss / total_goods_value) * 100 if total_goods_value > 0 else 0
    
    # Visualization
    if save_fig_path:
        # 标题显示参数变化
        dataset_name_with_params = f"Sensitivity Analysis\n{label} ({ALGORITHM_TYPE})"
        solVis(user_param, best_routes, sol_time, total_cost, dataset_name_with_params, POPOUT=False, save_path=save_fig_path)
    
    return {
        "label": label,
        "total_cost": total_cost,
        "emission": total_emission,
        "avg_dist": avg_dist,
        "loss_rate": loss_rate,
        "num_vehicles": len(best_routes)
    }

def run_sensitivity_analysis():
    # 使用随机分布的数据集 R101，更适合做灵敏度
    dataset = "dataset/R101.txt" 
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")

    print(f"=== Sensitivity Analysis using {ALGORITHM_TYPE} ===")

    # -------------------------------
    # 实验 1: 碳税灵敏度 (Carbon Tax)
    # -------------------------------
    print("\n>>> 1. Carbon Tax Sensitivity")
    tax_levels = [0.0, 0.05, 0.50, 1.0]
    for tax in tax_levels:
        print(f"Running for C_tax = {tax}...")
        label = f"Tax={tax}"
        fig_path = f"output/Sens_Tax_{tax}_{ALGORITHM_TYPE}.png"
        res = run_single_experiment(dataset, c_tax=tax, label=label, save_fig_path=fig_path)
        results.append(res)

    # -------------------------------
    # 实验 2: 时间窗灵敏度 (Hard vs Soft)
    # -------------------------------
    print("\n>>> 2. Time Window Sensitivity")
    # 0 = 硬时间窗, 30 = 允许迟到30分钟(软)
    tw_settings = [(0, "Hard TW"), (30, "Soft TW")]
    for lateness, name in tw_settings:
        print(f"Running for {name}...")
        fig_path = f"output/Sens_TW_{lateness}_{ALGORITHM_TYPE}.png"
        res = run_single_experiment(dataset, c_tax=0.05, max_lateness=lateness, label=name, save_fig_path=fig_path)
        results.append(res)

    # -------------------------------
    # 实验 3: 货损价值灵敏度 (P_fresh)
    # -------------------------------
    # 如果商品越贵，应该越快送到，可能会牺牲碳排放
    print("\n>>> 3. Freshness Price Sensitivity")
    prices = [10.0, 100.0] # 便宜蔬菜 vs 昂贵海鲜
    for p in prices:
        print(f"Running for P_fresh = {p}...")
        label = f"Price={p}"
        fig_path = f"output/Sens_Price_{p}_{ALGORITHM_TYPE}.png"
        res = run_single_experiment(dataset, p_fresh=p, label=label, save_fig_path=fig_path)
        results.append(res)

    # 输出汇总表格
    print(f"\n\n====== Sensitivity Results Summary ({ALGORITHM_TYPE}) ======")
    header = f"{'Scenario':<15} | {'Cost':<10} | {'Emission':<10} | {'Loss Rate':<10} | {'Vehicles':<8}"
    print(header)
    print("-" * 70)
    
    output_file = f"output/sensitivity_results_{ALGORITHM_TYPE}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"====== Sensitivity Results Summary ({ALGORITHM_TYPE}) ======\n")
        f.write(header + "\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            line = f"{r['label']:<15} | {r['total_cost']:<10.2f} | {r['emission']:<10.2f} | {r['loss_rate']:<10.2f}% | {r['num_vehicles']:<8}"
            print(line)
            f.write(line + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_sensitivity_analysis()