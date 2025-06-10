import random
from collections import deque
from copy import deepcopy
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple


class InitializationStrategy(Enum):
    """初始化策略枚举"""
    STANDARD = 1  # 标准拓扑排序初始化
    THRESHOLD = 2  # 阈值初始化策略（需提供预定义调度方案）


class RCPSP:
    """资源受限项目调度问题（RCPSP）"""

    def __init__(self, num_tasks: int, durations: List[int],
                 resource_requirements: List[List[int]],
                 resource_availabilities: List[int]):
        self.num_tasks = num_tasks
        self.durations = durations
        self.resource_requirements = resource_requirements
        self.resource_availabilities = resource_availabilities
        self.tasks = {}

    def add_task(self, task_id: int, duration: int,
                 predecessors: List[int], resources: List[int]):
        """添加任务"""
        self.tasks[task_id] = {
            'duration': duration,
            'predecessors': predecessors,
            'resources': resources
        }

    def get_successors(self, task_id: int) -> List[int]:
        """获取任务的后继任务"""
        return [t_id for t_id, task in self.tasks.items()
                if task_id in task['predecessors']]

    def detect_circular_dependencies(self) -> bool:
        """检测任务图中是否有循环依赖（DFS）"""
        visited = set()
        path = set()

        def dfs(node):
            visited.add(node)
            path.add(node)
            for neighbor in self.get_successors(node):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in path:
                    return True
            path.remove(node)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True
        return False

    def topological_sort(self) -> List[int]:
        """拓扑排序（Kahn算法）"""
        in_degree = {task_id: 0 for task_id in self.tasks}
        graph = {task_id: [] for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            for pred in task['predecessors']:
                graph[pred].append(task_id)
                in_degree[task_id] += 1

        queue = deque([task_id for task_id in self.tasks if in_degree[task_id] == 0])
        topo_order = []

        while queue:
            current = queue.popleft()
            topo_order.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(self.tasks):
            raise ValueError("任务图中存在循环依赖，无法进行拓扑排序！")
        return topo_order

    def validate_task_order(self, task_order: List[int]) -> bool:
        """验证任务顺序是否满足前驱约束"""
        scheduled = set()
        for task_id in task_order:
            if any(pred not in scheduled for pred in self.tasks[task_id]['predecessors']):
                return False
            scheduled.add(task_id)
        return True

    def serial_schedule_generation_scheme(self, task_order: Optional[List[int]] = None) -> Tuple[
        List[int], Dict[int, int]]:
        """串行调度生成方案（SGS）"""
        if task_order is None:
            task_order = self.topological_sort()

        if not self.validate_task_order(task_order):
            raise ValueError("任务顺序违反前驱约束！")

        start_times = {}
        resource_usage = [
            [0] * (sum(self.durations) + 1)
            for _ in range(len(self.resource_availabilities))
        ]

        for task_id in task_order:
            task = self.tasks[task_id]
            est = 0  # Earliest Start Time
            for pred in task['predecessors']:
                est = max(est, start_times[pred] + self.tasks[pred]['duration'])

            start_time = est
            while True:
                feasible = True
                for t in range(start_time, start_time + task['duration']):
                    for r in range(len(self.resource_availabilities)):
                        if (resource_usage[r][t] + task['resources'][r] >
                                self.resource_availabilities[r]):
                            feasible = False
                            break
                    if not feasible:
                        break

                if feasible:
                    break
                start_time += 1

            for t in range(start_time, start_time + task['duration']):
                for r in range(len(self.resource_availabilities)):
                    resource_usage[r][t] += task['resources'][r]

            start_times[task_id] = start_time

        return task_order, start_times

    def compute_makespan(self, start_times: Dict[int, int]) -> int:
        """计算总工期（Makespan）"""
        return max(st + self.tasks[task_id]['duration']
                   for task_id, st in start_times.items())


class TSABC_RCPSP:
    """禁忌搜索-人工蜂群算法（TSABC）"""

    def __init__(self,
                 project: RCPSP,
                 colony_size: int = 20,
                 max_iter: int = 100,
                 tabu_tenure: int = 5,
                 max_trials: int = 3,
                 init_strategy: InitializationStrategy = InitializationStrategy.STANDARD,
                 threshold_schedules = None):
        """
        参数:
        - project: RCPSP 实例
        - colony_size: 蜂群规模
        - max_iter: 最大迭代次数
        - tabu_tenure: 禁忌期限
        - max_trials: 最大尝试次数（侦察蜂触发条件）
        - init_strategy: 初始化策略（STANDARD / THRESHOLD）
        - threshold_schedule: 预定义的调度方案（仅当 init_strategy=THRESHOLD 时使用）
        """
        self.project = project
        self.colony_size = colony_size
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.max_trials = max_trials
        self.tabu_list = deque(maxlen=50)
        self.best_solution = None
        self.best_makespan = float('inf')
        self.init_strategy = init_strategy
        self.threshold_schedules = threshold_schedules

        # 验证阈值初始化参数
        if init_strategy == InitializationStrategy.THRESHOLD:
            for threshold_schedule in self.threshold_schedules:
                if not threshold_schedule or 'task_order' not in threshold_schedule or 'start_times' not in threshold_schedule:
                    raise ValueError("阈值初始化需要提供有效的预定义调度方案！")
                if not self._validate_threshold_schedule(threshold_schedule):
                    raise ValueError("提供的阈值调度方案无效！")

    def _validate_threshold_schedule(self, schedule: Dict) -> bool:
        """验证阈值调度方案的合法性"""
        try:
            # 检查前驱约束
            scheduled = set()
            for task_id in schedule['task_order']:
                if any(pred not in scheduled for pred in self.project.tasks[task_id]['predecessors']):
                    return False
                scheduled.add(task_id)

            # 检查资源约束
            max_time = max(st + self.project.tasks[task_id]['duration']
                           for task_id, st in schedule['start_times'].items())
            resource_usage = [
                [0] * (max_time + 1)
                for _ in range(len(self.project.resource_availabilities))
            ]

            for task_id, start_time in schedule['start_times'].items():
                task = self.project.tasks[task_id]
                for t in range(start_time, start_time + task['duration']):
                    for r in range(len(self.project.resource_availabilities)):
                        resource_usage[r][t] += task['resources'][r]
                        if resource_usage[r][t] > self.project.resource_availabilities[r]:
                            return False
            return True
        except:
            return False

    def solve(self) -> Optional[Dict]:
        population = self._initialize_population()

        for iteration in range(self.max_iter):
            # 雇佣蜂阶段
            for i in range(len(population)):
                neighbor = self._generate_neighbor(population[i])
                if neighbor:
                    neighbor_makespan = self._evaluate_schedule(neighbor)
                    if neighbor_makespan < population[i]['makespan']:
                        population[i] = neighbor
                        if neighbor_makespan < self.best_makespan:
                            self.best_solution = deepcopy(neighbor)
                            self.best_makespan = neighbor_makespan

            # 跟随蜂阶段（终极修复方案）
            valid_bees = [bee for bee in population if bee['makespan'] > 0]
            if not valid_bees:
                population = self._initialize_population()
                continue

            fitness_sum = sum(1 / bee['makespan'] for bee in valid_bees)
            probabilities = [(1 / bee['makespan']) / fitness_sum for bee in valid_bees]

            for _ in range(self.colony_size):
                # 直接选择 valid_bees 的索引
                selected_idx = random.choices(
                    range(len(valid_bees)),
                    weights=probabilities,
                    k=1
                )[0]

                # 获取 population 中的原始索引
                population_idx = [i for i, bee in enumerate(population)
                                  if bee['makespan'] > 0][selected_idx]

                neighbor = self._generate_neighbor(population[population_idx])
                if neighbor and neighbor['makespan'] < population[population_idx]['makespan']:
                    population[population_idx] = neighbor

            # 侦察蜂阶段
            for i in range(len(population)):
                if population[i]['trials'] >= self.max_trials:
                    population[i] = self._generate_standard_solution()

            print(f"Iter {iteration + 1}, Best: {self.best_makespan}")

        return self.best_solution

    def _initialize_population(self):
        """初始化种群（根据策略选择初始化方式）"""
        population = []
        while len(population) < self.colony_size:
            if self.init_strategy == InitializationStrategy.THRESHOLD:
                solution = self._generate_threshold_solution(len(population))
            else:
                solution = self._generate_standard_solution()

            if solution:
                population.append(solution)
        return population

    def _generate_standard_solution(self) -> Optional[Dict]:
        """标准拓扑排序初始化"""
        try:
            task_order = self.project.topological_sort()
            if random.random() < 0.7:  # 70%概率进行随机扰动
                task_order = self._constrained_shuffle(task_order)

            task_order, start_times = self.project.serial_schedule_generation_scheme(task_order)
            makespan = self.project.compute_makespan(start_times)
            return {
                'task_order': task_order,
                'start_times': start_times,
                'makespan': makespan,
                'trials': 0
            }
        except:
            return None

    def _generate_threshold_solution(self, index):
        """基于阈值策略的初始化"""
        try:
            # 使用预定义的调度方案作为基础
            base_order = self.threshold_schedules[index]['task_order']
            perturbed_order = self._constrained_shuffle(base_order.copy())

            # 重新计算时间
            task_order, start_times = self.project.serial_schedule_generation_scheme(perturbed_order)
            makespan = self.project.compute_makespan(start_times)

            return {
                'task_order': task_order,
                'start_times': start_times,
                'makespan': makespan,
                'trials': 0
            }
        except:
            return self._generate_standard_solution()  # 回退到标准策略

    def _generate_neighbor(self, solution: Dict) -> Optional[Dict]:
        """生成邻域解（带禁忌搜索）"""
        best_neighbor = None
        best_makespan = float('inf')
        best_move = None

        for _ in range(3):  # 生成3个候选解
            neighbor_order = solution['task_order'].copy()
            i, j = random.sample(range(len(neighbor_order)), 2)
            move = (neighbor_order[i], neighbor_order[j])

            if move in self.tabu_list:
                continue

            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]

            if self.project.validate_task_order(neighbor_order):
                try:
                    _, start_times = self.project.serial_schedule_generation_scheme(neighbor_order)
                    makespan = self.project.compute_makespan(start_times)

                    if makespan < best_makespan:
                        best_neighbor = {
                            'task_order': neighbor_order,
                            'start_times': start_times,
                            'makespan': makespan,
                            'trials': solution['trials'] + 1
                        }
                        best_makespan = makespan
                        best_move = move
                except:
                    continue

        if best_neighbor and best_move:
            self.tabu_list.append(best_move)
        return best_neighbor if best_neighbor else None

    def _is_allowed_move(self, current: Dict, neighbor: Dict) -> bool:
        """检查是否允许禁忌移动（藐视准则）"""
        return neighbor['makespan'] < self.best_makespan

    def _constrained_shuffle(self, task_order: List[int]) -> List[int]:
        """保持前驱约束的随机扰动"""
        for _ in range(len(task_order)):
            i, j = random.sample(range(len(task_order)), 2)
            new_order = task_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            if self.project.validate_task_order(new_order):
                task_order = new_order
        return task_order

    def _evaluate_schedule(self, solution: Dict) -> float:
        """评估调度方案"""
        if solution['makespan'] is None:
            try:
                _, start_times = self.project.serial_schedule_generation_scheme(solution['task_order'])
                solution['start_times'] = start_times
                solution['makespan'] = self.project.compute_makespan(start_times)
            except:
                solution['makespan'] = float('inf')
        return solution['makespan']


def validate_solution(project: RCPSP, task_order: List[int], task_start_times: Dict[int, int]) -> bool:
    """验证解决方案的合法性"""
    # 检查前驱约束
    for task_id in task_order:
        for pred in project.tasks[task_id]['predecessors']:
            if task_start_times[task_id] < task_start_times[pred] + project.tasks[pred]['duration']:
                print(f"前驱约束违反：任务{task_id}在任务{pred}完成前开始")
                return False

    # 检查资源约束
    max_time = max(st + project.tasks[task_id]['duration']
                   for task_id, st in task_start_times.items())
    resource_usage = [
        [0] * (max_time + 1)
        for _ in range(len(project.resource_availabilities))
    ]

    for task_id, start_time in task_start_times.items():
        task = project.tasks[task_id]
        for t in range(start_time, start_time + task['duration']):
            for r in range(len(project.resource_availabilities)):
                resource_usage[r][t] += task['resources'][r]
                if resource_usage[r][t] > project.resource_availabilities[r]:
                    print(f"资源{r}在时间{t}超限：{resource_usage[r][t]} > {project.resource_availabilities[r]}")
                    return False

    print("所有约束检查通过！")
    return True


def plot_schedule(schedule: List[int], project: RCPSP, task_start_times: Dict[int, int]):
    """绘制甘特图"""
    if schedule is None or task_start_times is None:
        print("无效参数，无法绘制甘特图！")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS.values())

    # 准备数据
    task_names = []
    start_times = []
    durations = []

    for task_id in schedule:
        task = project.tasks[task_id]
        task_names.append(f"Task {task_id}")
        start_times.append(task_start_times[task_id])
        durations.append(task['duration'])

    # 创建条形图
    for i in range(len(task_names)):
        ax.barh(
            task_names[i],
            durations[i],
            left=start_times[i],
            color=colors[i % len(colors)],
            edgecolor='black'
        )

        # 显示资源需求
        res_text = ",".join(map(str, project.tasks[schedule[i]]['resources']))
        ax.text(
            start_times[i] + durations[i] / 2,
            i,
            res_text,
            ha='center',
            va='center',
            color='white',
            fontsize=8
        )

    # 配置图表
    max_time = max(st + d for st, d in zip(start_times, durations))
    ax.set_xticks(range(0, max_time + 1))
    ax.set_xlabel('Time')
    ax.set_title('Project Schedule Gantt Chart')
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

from dataload import load_json_data, parse_task_info
from typing import List, Dict, Tuple, Optional
from params import *

def load_and_parse_data(test_json_path: str):
    """加载和解析测试集数据"""
    test_data = load_json_data(test_json_path)
    parsed_test_data = parse_task_info(test_data)
    return parsed_test_data

class Task:
    def __init__(self, id: int, duration: int, predecessors: List[int],
                 successors: List[int], resource_demands: List[int]):
        self.id = id
        self.duration = duration
        self.predecessors = predecessors
        self.successors = successors
        self.resource_demands = resource_demands


class Project:
    def __init__(self, tasks: List[Task], num_resources: int,
                 resource_availabilities: List[int]):
        self.tasks = tasks
        self.num_resources = num_resources
        self.resource_availabilities = resource_availabilities

def create_sample_project() -> Project:
    # 加载测试数据
    test_data = load_and_parse_data(TEST_JSON_PATH)  # 注意这里只传TEST_JSON_PATH
    Projects = []
    for sample in test_data:
        tasks = sample['tasks']
        res_limits = sample['res_limits']
        project = RCPSP(
            num_tasks=len(tasks),
            durations=[task['t'] for task in tasks],
            resource_requirements=[task['res'] for task in tasks],
            resource_availabilities=res_limits
        )
        for task in tasks:
            project.add_task(task['id'], task['t'], task['dep'], task['res'])

        Projects.append(project)
    return Projects

def OneStep():
    total_reward = 0
    projects = create_sample_project()
    for sample_id, project in enumerate(projects):
        # 检查循环依赖
        if project.detect_circular_dependencies():
            print("错误：任务图中存在循环依赖！")
        else:
            strategy = InitializationStrategy.STANDARD
            schedule = None
            tsabc = TSABC_RCPSP(
                project,
                colony_size=10,
                max_iter=50,
                tabu_tenure=5,
                max_trials=10,
                init_strategy=strategy,
                threshold_schedules=schedule
            )
            solution = tsabc.solve()

            if solution:
                print(f"sample_id: {sample_id}")
                print(f"最优解 (策略: {strategy}):")
                print(f"任务顺序: {solution['task_order']}")
                print(f"总工期: {solution['makespan']}")
                # validate_solution(project, solution['task_order'], solution['start_times'])
                total_reward += - solution['makespan']
            else:
                print(f"策略 {strategy} 未能找到有效解")

    avg_reward = total_reward / len(projects)
    print(f"测试平均奖励: {avg_reward:.2f}")

# 测试用例
if __name__ == "__main__":
    OneStep()