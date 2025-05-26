import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from heapq import heappush, heappop
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # 输入层从16改为 19 个特征，输出层为 3 个动作
        self.model = Linear_QNet(19, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.action_history = []
        self.window_size = 12
        self.right_turn_count = 0
        self.left_turn_count = 0
        self.last_move = None

    # 加了一个load
    def load_model(self, file_name='model.pth'):
        self.model.load(file_name)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # 重新初始化 trainer

    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        tail_left = tail.x < head.x
        tail_right = tail.x > head.x
        tail_up = tail.y < head.y
        tail_down = tail.y > head.y

        snake_body = set(game.snake[1:])
        obstacle_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_point = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                if (check_point in snake_body or
                        check_point.x < 0 or check_point.x >= game.w or
                        check_point.y < 0 or check_point.y >= game.h):
                    obstacle_count += 1
        obstacle_density = obstacle_count / 9

        # 添加未来几步的模拟
        future_collisions = [0] * 3  # [直行, 右转, 左转]
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i, action in enumerate(actions):
            next_head = self._get_next_head(game, action)
            if (next_head in snake_body or
                    next_head.x < 0 or next_head.x >= game.w or
                    next_head.y < 0 or next_head.y >= game.h):
                future_collisions[i] = 1
            elif len(game.snake) > 15:  # 当蛇较长时，检查是否会导致“自绕”
                danger_zone = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        check_point = Point(next_head.x + dx * BLOCK_SIZE, next_head.y + dy * BLOCK_SIZE)
                        if (check_point in snake_body or
                                check_point.x < 0 or check_point.x >= game.w or
                                check_point.y < 0 or check_point.y >= game.h):
                            danger_zone += 1
                if danger_zone >= 7:  # 周围空闲空间太少，可能导致“自绕”
                    future_collisions[i] = 1

        state = [
                    (dir_r and game.is_collision(point_r)) or
                    (dir_l and game.is_collision(point_l)) or
                    (dir_u and game.is_collision(point_u)) or
                    (dir_d and game.is_collision(point_d)),

                    (dir_u and game.is_collision(point_r)) or
                    (dir_d and game.is_collision(point_l)) or
                    (dir_l and game.is_collision(point_u)) or
                    (dir_r and game.is_collision(point_d)),

                    (dir_d and game.is_collision(point_r)) or
                    (dir_u and game.is_collision(point_l)) or
                    (dir_r and game.is_collision(point_u)) or
                    (dir_l and game.is_collision(point_d)),

                    dir_l,
                    dir_r,
                    dir_u,
                    dir_d,

                    game.food.x < game.head.x,
                    game.food.x > game.head.x,
                    game.food.y < game.head.y,
                    game.food.y > game.head.y,

                    tail_left,
                    tail_right,
                    tail_up,
                    tail_down,
                    obstacle_density
                ] + future_collisions  # 将未来碰撞信息添加到状态中

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def astar_pathfinding(self, game):
        start = (game.head.x // BLOCK_SIZE, game.head.y // BLOCK_SIZE)
        goal = (game.food.x // BLOCK_SIZE, game.food.y // BLOCK_SIZE)

        obstacles = set((pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE) for pt in game.snake)
        grid_w, grid_h = game.w // BLOCK_SIZE, game.h // BLOCK_SIZE

        open_list = []
        heappush(open_list, (0, 0, start, None))
        closed_set = set()
        came_from = {}

        while open_list:
            f, g, current, parent = heappop(open_list)
            if current in closed_set:
                continue

            closed_set.add(current)
            came_from[current] = parent

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                path.reverse()
                if len(path) > 1:
                    next_pos = path[1]
                    dx = next_pos[0] - start[0]
                    dy = next_pos[1] - start[1]
                    if game.direction == Direction.RIGHT:
                        if dx == 1:
                            return [1, 0, 0]  # 直行
                        elif dy == 1:
                            return [0, 1, 0]  # 右转
                        elif dy == -1:
                            return [0, 0, 1]  # 左转
                    elif game.direction == Direction.LEFT:
                        if dx == -1:
                            return [1, 0, 0]  # 直行
                        elif dy == -1:
                            return [0, 1, 0]  # 右转
                        elif dy == 1:
                            return [0, 0, 1]  # 左转
                    elif game.direction == Direction.DOWN:
                        if dy == 1:
                            return [1, 0, 0]  # 直行
                        elif dx == -1:
                            return [0, 1, 0]  # 右转
                        elif dx == 1:
                            return [0, 0, 1]  # 左转
                    elif game.direction == Direction.UP:
                        if dy == -1:
                            return [1, 0, 0]  # 直行
                        elif dx == 1:
                            return [0, 1, 0]  # 右转
                        elif dx == -1:
                            return [0, 0, 1]  # 左转
                return [1, 0, 0]  # 默认直行

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (neighbor in closed_set or
                        neighbor in obstacles or
                        neighbor[0] < 0 or neighbor[0] >= grid_w or
                        neighbor[1] < 0 or neighbor[1] >= grid_h):
                    continue

                new_g = g + 1
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                min_dist_to_body = min(abs(neighbor[0] - bx) + abs(neighbor[1] - by)
                                       for bx, by in obstacles) if obstacles else 10
                safety_bonus = min_dist_to_body * 0.2
                f = new_g + h - safety_bonus

                heappush(open_list, (f, new_g, neighbor, current))

        # 未找到食物时，返回安全方向
        head = game.head
        if game.direction == Direction.RIGHT:
            if head.x >= game.w - 2 * BLOCK_SIZE:
                return [0, 1, 0]  # 靠近右墙，右转
            return [1, 0, 0]
        elif game.direction == Direction.LEFT:
            if head.x < 2 * BLOCK_SIZE:
                return [0, 1, 0]  # 靠近左墙，右转
            return [1, 0, 0]
        elif game.direction == Direction.DOWN:
            if head.y >= game.h - 2 * BLOCK_SIZE:
                return [0, 1, 0]  # 靠近下墙，右转
            return [1, 0, 0]
        elif game.direction == Direction.UP:
            if head.y < 2 * BLOCK_SIZE:
                return [0, 1, 0]  # 靠近上墙，右转
            return [1, 0, 0]

    def _get_next_head(self, game, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        x = game.head.x
        y = game.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)

    def _will_trap_self(self, game, next_head):
        snake_body = set(game.snake[1:])
        if next_head in snake_body:
            return True

        if len(game.snake) > 15:
            head_x, head_y = next_head.x, next_head.y
            danger_zone = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_point = Point(head_x + dx * BLOCK_SIZE, head_y + dy * BLOCK_SIZE)
                    if (check_point in snake_body or
                            check_point.x < 0 or check_point.x >= game.w or
                            check_point.y < 0 or check_point.y >= game.h):
                        danger_zone.add((check_point.x, check_point.y))
            if len(danger_zone) >= 7:
                return True
        return False

    def get_action(self, state, game):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = torch.argmax(prediction).item()

        attempted_move = [0, 0, 0]
        attempted_move[move] = 1
        move_id = move

        self.action_history.append(move_id)
        if len(self.action_history) > self.window_size:
            self.action_history.pop(0)

        if self.last_move is not None:
            if (self.last_move == 1 and move_id == 2) or (self.last_move == 2 and move_id == 1):
                self.right_turn_count = 0
                self.left_turn_count = 0
            elif move_id == 1:
                self.right_turn_count += 1
                self.left_turn_count = 0
            elif move_id == 2:
                self.left_turn_count += 1
                self.right_turn_count = 0
            elif move_id == 0:
                self.right_turn_count = 0
                self.left_turn_count = 0
        self.last_move = move_id

        if len(game.snake) > 25:
            turn_limit = 1
        else:
            turn_limit = 3

        if move_id == 1 and self.right_turn_count >= turn_limit:
            available_moves = [[1, 0, 0], [0, 0, 1]]
            available_indices = [0, 2]
            q_values = [prediction[i].item() for i in available_indices]
            best_move_idx = available_indices[np.argmax(q_values)]
            final_move = [0, 0, 0]
            final_move[best_move_idx] = 1
        elif move_id == 2 and self.left_turn_count >= turn_limit:
            available_moves = [[1, 0, 0], [0, 1, 0]]
            available_indices = [0, 1]
            q_values = [prediction[i].item() for i in available_indices]
            best_move_idx = available_indices[np.argmax(q_values)]
            final_move = [0, 0, 0]
            final_move[best_move_idx] = 1
        else:
            final_move = attempted_move

        next_head = self._get_next_head(game, final_move)
        if (next_head.x < BLOCK_SIZE or next_head.x >= game.w - BLOCK_SIZE or
                next_head.y < BLOCK_SIZE or next_head.y >= game.h - BLOCK_SIZE or
                self._will_trap_self(game, next_head)):
            final_move = self.astar_pathfinding(game)

        return final_move


def plot_extended(scores, mean_scores, wall_collisions, self_collisions, timeouts):
    plt.ion()
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='orange')
    plt.title('Scores over Games')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(wall_collisions, label='Wall Collisions', color='red')
    plt.plot(self_collisions, label='Self Collisions', color='green')
    plt.plot(timeouts, label='Timeouts', color='purple')
    plt.title('Death Causes over Games')
    plt.xlabel('Number of Games')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.pause(0.001)

def train(load_model=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    wall_collisions = []
    self_collisions = []
    timeouts = []
    wall_count = 0
    self_count = 0
    timeout_count = 0

    agent = Agent()
    if load_model:
        agent.load_model()

    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, game)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 计算距离奖励
        dist_old = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
        dist_new = abs(game.snake[0].x - game.food.x) + abs(game.snake[0].y - game.food.y)
        if dist_new < dist_old:
            reward += 2
        elif dist_new > dist_old:
            reward -= 1

        # 惩罚“自绕”风险
        if len(game.snake) > 15:
            head = game.snake[0]
            danger_zone = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_point = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                    if (check_point in set(game.snake[1:]) or
                            check_point.x < 0 or check_point.x >= game.w or
                            check_point.y < 0 or check_point.y >= game.h):
                        danger_zone += 1
            if danger_zone >= 7:
                reward -= 0.5  # 周围空闲空间过少，给予惩罚

        snake_length = len(game.snake)
        if snake_length > 20:
            safety_factor = 1.2 - (snake_length / 100)
            if reward == 10:
                reward = 10 * safety_factor
            elif reward == -10:
                reward = -10 * (1.5 - safety_factor)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            head = game.head
            if (head.x >= game.w - BLOCK_SIZE or head.x < 0 or
                    head.y >= game.h - BLOCK_SIZE or head.y < 0):
                wall_count += 1
            elif head in game.snake[1:]:
                self_count += 1
            elif game.frame_iteration > 100 * len(game.snake):
                timeout_count += 1

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record,
                  'Wall:', wall_count, 'Self:', self_count, 'Timeout:', timeout_count)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            wall_collisions.append(wall_count)
            self_collisions.append(self_count)
            timeouts.append(timeout_count)

            plot_extended(plot_scores, plot_mean_scores, wall_collisions, self_collisions, timeouts)


if __name__ == '__main__':
    # True 以加载模型，或者 False 以从头训练
    train(load_model=True)