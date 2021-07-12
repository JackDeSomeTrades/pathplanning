"""
Path planning with RRTConnect on a grid environment
This is an extensible sampling based algorithm for planning between two states in the environment. The environment can
be either a 4-connected grid (N-S-E-W agent movement) or 8-connected (N-S-E-W-NE-NW-SE-SW agent movement). The RRT is
extremely useful when exploring in an unknown region of the environment. The RRT will always return a path between the
start and the stop positions, however the returned path may not always be optimal (Other variants exist such as RRT*
that return the optimal path). The RRT variant presented here is RRTConnect[1] - a bidirectional RRT that begins exploration
from both the start and the stop positions and extends until the two trees meet at a node. This variant is
quicker in time to converge to a viable solution and takes fewer iterations to arrive at a solution compared to the vanilla
RRT variant.

The RRT works as follows -
The exploration trees on both sides are initalised and a random node is generated within the bounds of the environment.
Then the nearest neighbor to this random node is calculated and a new node is created in the direction of this random node
and the tree is extended, after checking if this new node is viable (no collision, not inside obstacle, etc.). This is
performed on both exploration trees until they both explore the same node. This is the termination point and a path exists
in the environment between the start and stop positions.

-------
[1] Kuffner, J. J., & LaValle, S. M. (2000, April). RRT-connect: An efficient approach to single-query path planning.
    In Proceedings 2000 ICRA. Millennium Conference. IEEE International Conference on Robotics and Automation.(Vol. 2, pp. 995-1001). IEEE.


Implementation:

- Assume a 4-connected grid.
- All cell states and coordinates are reachable.


@author: Pavan Vasishta, PhD
"""
import numpy as np
import math


np.random.seed(42)

grid = [[1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0]]

grid = np.array(grid)

start_xy = (4, 0)
stop_xy  = (0, 4)


class Node:
    """
    Node class contains data on the spatial positon of the node and the parent of the node in the random tree.
    """
    def __init__(self, node):
        self.parent = None
        self.x = node[0]
        self.y = node[1]

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False


class Map:
    """
    Map class for encapsulating the grid data. Contains data on the occupancy of the grid and corresponding co-ordinates
    :param grid: 2-D numpy matrix of the environment. Each cell should contain occupancy data (0 or 1)
    """
    def __init__(self, grid):
        self.grid = grid
        res = np.where(self.grid == 1)
        self.obstacles = set(list(zip(res[0], res[1])))  # get the co-ordinates of the obstacles in the grid (defined as
                                                    # grid cell value == 1)
        self.step = 1
        self.bounds = np.shape(grid)


class RRTConnect:
    """
    Implementation of the RRTConnect variant of RRTs; Bidirectional RRT with termination when both trees explore the
    same node.
    :param map: Map instance, contains information on the grid and the environment.
    :param start: tuple containing x,y position of the start position on the grid
    :param stop: tuple containing x,y position of stop position on the grid
    :param MAX_ITER: int for the maximum iteration value of exploration of the grid by the bidirectional tree. Value
                     should increase as the size of the grid increases.
    """
    def __init__(self, map, start, stop, MAX_ITER = 100):
        self.environment = map
        self.step_length = self.environment.step

        self.start = Node(start)
        self.stop = Node(stop)

        self.forward_explore = [self.start]  # initialise forward exploration tree
        self.backward_explore = [self.stop]  # intitalise backward exploration tree

        self.MAX_ITER = MAX_ITER

    def plan(self):
        for i in range(self.MAX_ITER):
            q_rand_node = self.generate_random_node()
            q_nearest = self.find_nearest_neighbour(self.forward_explore, q_rand_node)
            q_new = self.make_new_state(q_nearest, q_rand_node)
            if q_new is not None:
                self.forward_explore.append(q_new)   # extend forward
                q_b_nearest = self.find_nearest_neighbour(self.backward_explore, q_new)
                q_b_new = self.make_new_state(q_b_nearest, q_new)
                if q_b_new is not None:
                    self.backward_explore.append(q_b_new)  # extend backward
                    while True:
                        q_b_new_prime = self.make_new_state(q_b_new, q_new)
                        if q_b_new_prime is not None:
                            self.backward_explore.append(q_b_new_prime)
                            q_b_new = self.swap(q_b_new, q_b_new_prime)
                        else:
                            break
                        if q_b_new == q_new:
                            # termination condition reached
                            break
                else:
                    continue
                if q_new == q_b_new:
                    # Path has been found between the start and stop positions
                    path = self.return_path(q_new, q_b_new)
                    return path
            else:
                continue

    def return_path(self, q_fwd, q_bwd):
        """
        :param q_fwd: forward exploration tree, latest node
        :param q_bwd: backward exploration tree, latest node
        :return: path: list of nodes
        """
        fwd_pth = [(q_fwd.x, q_fwd.y)]
        bwd_pth = [(q_bwd.x, q_bwd.y)]
        q_ = q_fwd
        _q = q_bwd
        while q_.parent is not None:
            q_ = q_.parent
            fwd_pth.append((q_.x, q_.y))
        while _q.parent is not None:
            _q = _q.parent
            bwd_pth.append((_q.x, _q.y))

        fwd_pth = list(reversed(fwd_pth))[:-1]  # To remove repated nodes from forward and backward tree since they overlap
        path = fwd_pth + bwd_pth

        return path

    def swap(self, q_old, q_prime):
        q_new = Node((q_prime.x, q_prime.y))
        q_new.parent = q_old

        return q_new

    def make_new_state(self, q_1, q_2):
        # Make new state around the direction of the random generated node with a one step motion
        angle = math.atan2((q_2.x - q_1.x), (q_2.y - q_1.y))   # to account for the x and y flip between numpy and real life

        # Assuming a 4-connected grid for agent motion. If 8-connected, the angles do not need to be forced as below.
        if -math.pi/4 < angle <= math.pi/4:
            angle = 0
        elif math.pi/4 < angle <= 3 * math.pi/4:
            angle = math.pi/2
        elif 3 * math.pi/4 < angle <= -3 * math.pi/4:
            angle = math.pi
        elif -3 * math.pi/4 < angle <= - math.pi/4:
            angle = -math.pi/2

        new_x, new_y = q_1.x + round(math.sin(angle)), q_1.y + round(math.cos(angle))  # assuming step length is 1

        nd = {(new_x, new_y)}

        # check if new node is an obstacle or not
        if not self.environment.obstacles.intersection(nd) and \
            new_x < self.environment.bounds[0] and new_y < self.environment.bounds[1]:
            q_new = Node((new_x, new_y))
            q_new.parent = q_1
        else:
            q_new = None

        return q_new

    @staticmethod
    def find_nearest_neighbour(node_tree, new_node):
        # check the euclidean distance between each of the nodes in the tree and the newest node.
        # return the co-ordinates of node with the smallest distance from the tested node.
        dsts = []
        for node in node_tree:
            dist = math.sqrt((new_node.x - node.x)**2 + (new_node.y - node.y)**2)
            dsts.append(dist)
        dsts = np.array(dsts)
        nearest_node = node_tree[np.argmin(dsts)]
        return nearest_node

    def generate_random_node(self):
        # return a random node sampled from the environment.
        return Node((np.random.randint(0, self.environment.bounds[0]), np.random.randint(0, self.environment.bounds[1])))


def myPathPlanning (grid, start, stop):
    map = Map(grid)
    rrt_planner = RRTConnect(map, start, stop)
    path = rrt_planner.plan()
    print("The planned path between the start and stop points can be found as:", path)


if __name__ == '__main__':
    myPathPlanning(grid, start_xy, stop_xy)
