import numpy as np
import time
import os
from branchBound import BranchAndBound
from heuristicGA import GeneticAlgorithm  # [新增] 导入你的启发式算法类
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis

# ==========================================
# [配置区] 选择你要使用的算法
# 选项: "GA" (启发式-遗传算法) 或 "BP" (精确-分支定价)
# ==========================================
ALGORITHM_TYPE = "GA" 
# ALGORITHM_TYPE = "BP"  # 如果想跑精确算法，取消这行注释，注释上一行

def run_scale_experiment(datasetPath, num_customers, label="", save_fig_path=None):
    # 1. 初始化通用参数
    user_param = ParamsVRP()
    # 限制客户数量 (max_customers) 用于规模测试
    user_param.init_params(datasetPath, max_customers=num_customers)
    
    best_routes = []
    start_time = time.time()
    
    # ==========================================
    # 分支 1: 运行 启发式算法 (GA)
    # ==========================================
    if ALGORITHM_TYPE == "GA":
        print(f"[{label}] Running Genetic Algorithm...")
        
        # 动态调整参数：规模越大，迭代次数越多
        gens = 200 if num_customers > 50 else 100
        pop_size = 100
        
        ga = GeneticAlgorithm(user_param, pop_size=pop_size, generations=gens)
        # GA 返回的是 (routes, cost, time)
        best_routes, _, _ = ga.run()
        
    # ==========================================
    # 分支 2: 运行 精确算法 (B&P)
    # ==========================================
    elif ALGORITHM_TYPE == "BP":
        print(f"[{label}] Running Branch & Price (Exact)...")
        
        # B&P 需要构建初始可行解 (Depot -> Client -> Depot)
        bp = BranchAndBound()
        init_routes = []
        for i in range(user_param.nbclients - 2):
            path = [0, i + 1, user_param.nbclients - 1]
            # 调用 paramsVRP 的计算函数，确保成本定义一致
            route_cost = user_param.calculate_actual_cost(path)
            route = Route(path=path, cost=route_cost, Q=1.0)
            init_routes.append(route)
            
        # 运行 B&P
        bp.bb_node(user_param, init_routes, None, best_routes, 0)
        
    else:
        raise ValueError("Unknown ALGORITHM_TYPE. Please choose 'GA' or 'BP'.")

    end_time = time.time()
    sol_time = end_time - start_time
    
    # ==========================================
    # 3. 统一指标统计 (通用逻辑)
    # 无论是 GA 还是 BP，最后都返回 Route 对象列表，所以统计逻辑是一样的
    # ==========================================
    total_cost = 0
    total_dist = 0
    total_emission = 0 
    total_fresh_loss = 0
    
    for route in best_routes:
        # 确保使用最新的 Route cost (如果是 B&P 跑出来的，cost 属性已经是准确的)
        # 如果是 GA 跑出来的，cost 也是准确的
        total_cost += route.get_cost()
        
        path = route.get_path()
        current_time = 0.0
        
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            
            # 处理距离: 如果 B&P 修改了矩阵导致无穷大，回退到 dist_base
            d_ij = user_param.dist[i][j]
            if d_ij >= user_param.verybig / 100:
                d_ij = user_param.dist_base[i][j]
                
            total_dist += d_ij
            
            # 碳排放计算: 距离 * 油耗 * 排放因子
            emission = d_ij * user_param.rho_avg * user_param.eta_CO2
            total_emission += emission
            
            # 时间更新 (用于计算货损)
            if k == 0:
                departure_at_i = 0
            else:
                start_service_at_i = max(current_time, user_param.a[i])
                departure_at_i = start_service_at_i + user_param.s[i]
            
            arrival_at_j = departure_at_i + user_param.ttime[i][j]
            current_time = arrival_at_j
            
            # 货损计算 (排除 Depot)
            if j != 0 and j != user_param.nbclients - 1:
                freshness_cost = user_param.P_fresh * user_param.d[j] * user_param.theta * current_time
                total_fresh_loss += freshness_cost

    # 平均路径长度
    avg_dist = total_dist / len(best_routes) if len(best_routes) > 0 else 0
    
    # 可视化保存
    if save_fig_path:
        # 在图片标题中带上算法名字
        dataset_name_with_params = f"C110_1 (N={num_customers})\nAlgorithm: {ALGORITHM_TYPE}"
        solVis(user_param, best_routes, sol_time, total_cost, dataset_name_with_params, POPOUT=False, save_path=save_fig_path)
    
    return {
        "scale": num_customers,
        "method": ALGORITHM_TYPE,
        "total_cost": total_cost,
        "time_sec": sol_time,
        "emission": total_emission,
        "num_vehicles": len(best_routes),
        "avg_dist": avg_dist
    }

def run_scale_analysis():
    # 默认数据集
    dataset = "dataset/C110_1.TXT"
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # ------------------------------------
    # 设置你要测试的规模列表
    # ------------------------------------
    # 如果是用 B&P，建议只跑 [15, 25, 30]
    # 如果是用 GA，可以跑 [25, 50, 100]
    if ALGORITHM_TYPE == "BP":
        scales = [15, 25] 
        print("Warning: Running B&P on large scales may take forever.")
    else:
        scales = [25, 50, 100]
    
    print(f"=== Experiment 1: Scale Analysis using {ALGORITHM_TYPE} ===")
    
    for n in scales:
        print(f"\n--- Processing N = {n} ---")
        fig_path = f"output/Scale_N{n}_{ALGORITHM_TYPE}.png"
        
        try:
            res = run_scale_experiment(dataset, num_customers=n, label=f"N={n}", save_fig_path=fig_path)
            results.append(res)
            print(f"Done. Cost={res['total_cost']:.2f}, Time={res['time_sec']:.2f}s")
        except Exception as e:
            print(f"Failed for N={n}: {e}")

    # 输出表格
    print(f"\n\n====== Scale Experiment Results ({ALGORITHM_TYPE}) ======")
    header = f"{'N':<5} | {'Cost':<10} | {'Time(s)':<10} | {'Emission':<10} | {'Vehicles':<8}"
    print(header)
    print("-" * 60)
    
    output_file = f"output/scale_results_{ALGORITHM_TYPE}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"====== Scale Experiment Results ({ALGORITHM_TYPE}) ======\n")
        f.write(header + "\n")
        f.write("-" * 60 + "\n")
        
        for r in results:
            line = f"{r['scale']:<5} | {r['total_cost']:<10.2f} | {r['time_sec']:<10.2f} | {r['emission']:<10.2f} | {r['num_vehicles']:<8}"
            print(line)
            f.write(line + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_scale_analysis()