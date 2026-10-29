import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

class IslandCityEnv(gym.Env):
    """
    Custom Gymnasium Environment for Island City Building with 5-channel observations
    and multi-tile Action Masking.
    """
    metadata = {"render_modes": []}

    def __init__(self, height=12, width=12, max_steps=100):
        super().__init__()
        self.H = height
        self.W = width
        self.max_steps = max_steps
        self.market_range = 4
        
        # Footprint sizes: (height, width)
        self.BUILDING_SIZES = {
            0: (1, 1), # Road 
            1: (1, 1), # House 
            2: (1, 1)  # Market 
        }
        
        # Action Space: 3 buildings * H * W
        self.n_buildings = 3
        self.action_space = spaces.Discrete(self.n_buildings * self.H * self.W)
        
        # Observation Space: (5, H, W)
        # Ch 0: Land Mask | Ch 1: Roads | Ch 2: Houses | Ch 3: Markets | Ch 4: Connectivity map
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5, self.H, self.W), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_score = 0.0
        
        # 1 = Land, 0 = Water
        self.land_mask = np.ones((self.H, self.W), dtype=np.float32)
        
        # Grid containing building IDs (0=empty, 1=road, 2=house, 3=market)
        self.building_grid = np.zeros((self.H, self.W), dtype=np.int8)
        
        # Unique IDs map to track multi-tile house/market instances
        self.instance_grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.next_instance_id = 1
        self.instance_dict = {}
        # Road and house distance map 
        self.dist_map = np.full((self.H, self.W), 99.9, dtype=np.float32)

        self.action_mask = np.zeros(self.action_space.n, dtype=bool)
        # action mask for roads only
        self.exlusive_road_builds = set() # only for tiles adjacent to market
        self.allowed_buidling_builds = set()
        self.disallowed_building_build = set()

        # SEEDED START: Pre-place 1 Market and 1 adjacent Road in the center
        mid_y, mid_x = self.H // 2 - 1, self.W // 2 - 1
        self._building_agent(mid_y, mid_x, 2) # Market (1x1)
        
        #self._update_distances_and_score() 
        return self._get_obs(), {"action_mask": self.action_masks()}

    def _get_obs(self):
        """Constructs the 5-channel observation tensor (5, H, W)."""
        obs = np.zeros((5, self.H, self.W), dtype=np.float32)
        obs[0] = self.land_mask
        obs[1] = (self.building_grid == 0).astype(np.float32)
        obs[2] = (self.building_grid == 1).astype(np.float32)
        obs[3] = (self.building_grid == 2).astype(np.float32)
        
        # Normalize distance channel to [0, 1] relative to max scoring range (self.market_range)
        road_only_dist_map = np.where(self.building_grid == 1, self.dist_map, 0.0)
        obs[4] = self._normalize_house_reward(road_only_dist_map) # pass distance map filtered down to road tiles only
        return obs
    def _building_agent(self, y, x, btype) -> float: # returns change in score
        self._place_building_tiles(y, x, btype)
        score_change = 0
        if btype == 0: # road
            score_change -= 0
            min_dist_to_market = self.market_range*999
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    if self.building_grid[ny, nx] == 1: #is next to other road:
                        min_dist_to_market = min(min_dist_to_market, self.dist_map[ny, nx] + 1)
                    elif self.building_grid[ny, nx] == 3: #is next to market
                        min_dist_to_market = 0
                    elif self.building_grid[ny, nx] == 0: #possible next road or building
                        self.allowed_buidling_builds.add((ny,nx))
            self.dist_map[y, x] = min_dist_to_market
            queue = deque()
            queue.append((y, x))
            # BFS over roads that have new shortest connection to market
            while queue:
                y, x = queue.popleft()
                dist = self.dist_map[y, x]
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]: 
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W :
                        cur_building = self.building_grid[ny, nx] 
                        if cur_building == 1 and self.dist_map[ny, nx] > dist + 1: # check if neighbour roads are now closer to market
                            queue.append((ny, nx))
                            self.dist_map[ny, nx] = dist + 1
                        elif cur_building == 2 and self.instance_dict[self.instance_grid[ny, nx]] > dist: # check if neighbour houses are now closer to market
                            score_change += self._normalize_house_reward(dist) - self._normalize_house_reward(self.instance_dict[self.instance_grid[ny, nx]])
                            self.instance_dict[self.instance_grid[ny, nx]] = dist

        elif btype == 1: #house
            bh, bw = self.BUILDING_SIZES[btype]
            neighbors = self._get_orthogonal_neighbors((self.building_grid == 1),y,x,bw,bh)
            min_dist_to_market = self.market_range*999
            for nx, ny in neighbors:
                if self.building_grid[ny, nx] == 1:
                    min_dist_to_market = min(min_dist_to_market, self.dist_map[ny, nx])
            self.instance_dict[self.next_instance_id - 1] = min_dist_to_market
            score_change += self._normalize_house_reward(min_dist_to_market)
        elif btype == 2: # market
            bh, bw = self.BUILDING_SIZES[btype]
            empty_neighbours = self._get_orthogonal_neighbors(self.building_grid == 0, y, x, bw, bh)
            for nx, ny in empty_neighbours:
                self.exlusive_road_builds.add((ny,nx))
        return score_change
    
    def action_masks(self) -> np.ndarray:
        self._update_action_mask()
        return self._get_action_mask()
    
    def _get_action_mask(self):
        return self.action_mask
    def _update_action_mask(self):
        mask = self._get_action_mask()
        road_or_market = (self.building_grid == 1) | (self.building_grid == 3)
        road_mask = (self.building_grid == 1)
        house_mask = (self.building_grid == 2)
        
        for btype in range(self.n_buildings): # new builds
            bh, bw = self.BUILDING_SIZES[btype]
            yx_indexes = self.allowed_buidling_builds if btype != 0 else self.allowed_buidling_builds | self.exlusive_road_builds
            for y, x in yx_indexes:
            
                idx = btype * (self.H * self.W) + y * self.W + x
                
                # 1. Boundary & Land check
                if y + bh > self.H or x + bw > self.W:
                    continue
                if not np.all(self.land_mask[y:y+bh, x:x+bw] == 1):
                    continue
                    
                footprint = self.building_grid[y:y+bh, x:x+bw]
                
                
                # Must be completely empty land to build
                if not np.all(footprint == 0):
                    continue
                
                
                if btype == 0: # ROAD: must touch an existing road or market and be in range of market; must be shortest path to market
                    market_y, market_x = self.H // 2 - 1, self.W // 2 - 1 # needs change for multiple markets
                    is_shortest_path = False
                    orthogonal_neighbors = self._get_orthogonal_neighbors(road_or_market, y, x, bw, bh)
                    for nx, ny in orthogonal_neighbors:
                        far_x, far_y = 2*(nx - x) + x, y + 2*(ny - y) #prevention from creating ending roads, if there is house on the other side of the road there can only be building on this side
                        if 0 <= far_x < self.W and 0 <= far_y < self.H and self.building_grid[far_y, far_x] == 2:
                            mask[idx] = False
                            break
                        if self.dist_map[ny, nx] <  2*self.market_range and self.dist_map[ny, nx] + 1 <= abs(y - market_y) + abs(x - market_x) + 3:
                            mask[idx] = True
                        elif self.building_grid[ny, nx] == 3:
                            mask[idx] = True
                        
                        
                elif btype == 1: # HOUSE: must touch at least one road
                    orthogonal_roads = self._get_orthogonal_neighbors(road_mask, y, x, bw, bh)
                    if len(orthogonal_roads) > 0:
                        will_be_deadend = False
                        is_l_crossroad = False
                        
                        for nx, ny in orthogonal_roads:
                            far_x, far_y = 2*(nx - x) + x, y + 2*(ny - y)
                            if 0 <= far_x < self.W and 0 <= far_y < self.H and self.building_grid[far_y, far_x] == 1:
                                will_be_deadend = True
                                break
                        if not will_be_deadend:
                            mask[idx] = True
                        
                elif btype == 2: # MARKET: can be placed anywhere valid
                    pass
                    #mask[idx] = True


        for btype in range(self.n_buildings):
            for y, x in self.disallowed_building_build:
                idx = btype * (self.H * self.W) + y * self.W + x
                mask[idx] = False
    def _normalize_house_reward(self, distance, fields = None):
      # Restrict (distance - market_range) to values between 0 and market_range
      clipped_dist = np.clip(
          distance - self.market_range, 0, self.market_range
      )
      return 1.0 - (clipped_dist / self.market_range)

    def step(self, action):
        self.current_step += 1
        
        # Decode action index -> (btype, y, x)
        btype = action // (self.H * self.W)
        rem = action % (self.H * self.W)
        y, x = rem // self.W, rem % self.W
        
        bh, bw = self.BUILDING_SIZES[btype]
        
        # Execute action
        reward = self._building_agent(y, x, btype)
            
        # Re-evaluate pathfinding & score
        old_score = self.total_score
        #new_score = self._update_distances_and_score()
        
        # Delta reward calculation (-0.01 step penalty to discourage endless loops)
        #reward = (new_score - old_score) - 0.01
        #reward = (new_score - old_score)
        new_score = old_score + reward
        self.total_score = new_score

        mask = self.action_masks()
        terminated = not np.any(mask)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"action_mask": mask}

    def _place_building_tiles(self, y, x, btype):
        bh, bw = self.BUILDING_SIZES[btype]
        # Convert action building ID -> grid building ID
        grid_id = btype + 1 
        for in_y in range(y, y + bh): #iterate thorugh tiles building will occupy
            for in_x in range(x, x+bw):
                self.disallowed_building_build.add((in_y, in_x))
                # self.exlusive_road_builds.discard((in_y, in_x))
                # self.allowed_buidling_builds.discard((in_y, in_x))
        self.building_grid[y:y+bh, x:x+bw] = grid_id
        if btype in (1, 2):
            self.instance_grid[y:y+bh, x:x+bw] = self.next_instance_id
            self.next_instance_id += 1
    def _has_orthogonal_neighbor(self, grid, y, x, bw, bh):
        """Checks if anx orthogonal neighbor cell is True, short-circuiting on the first hit."""
        # Top edge
        if x > 0 and np.anx(grid[x - 1, y : min(self.W, y + bw)]):
            return True
        # Bottom edge
        if x + bh < self.H and np.anx(grid[x + bh, y : min(self.W, y + bw)]):
            return True
        # Left edge
        if y > 0 and np.anx(grid[x : min(self.H, x + bh), y - 1]):
            return True
        # Right edge
        if y + bw < self.W and np.anx(grid[x : min(self.H, x + bh), y + bw]):
            return True

        return False
    def _get_orthogonal_neighbors(self, grid, y, x, bw, bh) -> list:
        """Returns a list of (y, x) coordinates for all orthogonal neighbor cells that are True."""
        neighbors = []

        # Top edge
        if x > 0:
            y_end = min(self.W, y + bw)
            row = grid[x - 1, y:y_end]
            for offset, val in enumerate(row):
                if val:
                    neighbors.append((y + offset, x - 1))

        # Bottom edge
        if x + bh < self.H:
            y_end = min(self.W, y + bw)
            row = grid[x + bh, y:y_end]
            for offset, val in enumerate(row):
                if val:
                    neighbors.append((y + offset, x + bh))

        # Left edge
        if y > 0:
            x_end = min(self.H, x + bh)
            col = grid[x:x_end, y - 1]
            for offset, val in enumerate(col):
                if val:
                    neighbors.append((y - 1, x + offset))

        # Right edge
        if y + bw < self.W:
            x_end = min(self.H, x + bh)
            col = grid[x:x_end, y + bw]
            for offset, val in enumerate(col):
                if val:
                    neighbors.append((y + bw, x + offset))

        return neighbors
    def render(self):
        """Renders the current city layout to the terminal using ASCII/Emojis."""
        symbols = {
            0: " 🟩",  # Empty Land
            1: " ⬛",  # Road
            2: " 🏠",  # House (2x2)
            3: " 🏢"   # Market (3x3)
        }
        
        print(f"\n--- Step: {self.current_step}/{self.max_steps} | Score: {self.total_score:.2f} ---")
        #RED = "\033[31m"
        #GREEN = "\033[32m"
        #RESET = "\033[0m"
        for y in range(self.H):
            row_str = ""
            for x in range(self.W):
                btype = self.building_grid[y, x]
                if btype == 1:
                    s = str(int(self.dist_map[y, x]))
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                elif btype == 2:
                    s = str(int(self.instance_dict[self.instance_grid[y, x]]))
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                else:
                    row_str += "##"
                row_str += "|"
            print(row_str)
        for y in range(self.H):
            row_str = ""
            for x in range(self.W):
                btype = self.building_grid[y, x]
                row_str += symbols.get(btype, " ? ")
            print(row_str)
    def _update_distances_and_score(self) -> float:
        """BFS Pathfinding over roads to compute house score decay."""
        self.dist_map.fill(99.0)
        
        # 1. Identify road tiles adjacent to Markets
        queue = deque()
        market_mask = (self.building_grid == 3)
        road_mask = (self.building_grid == 1)
        
        for y in range(self.H):
            for x in range(self.W):
                if road_mask[y, x]:
                    # Check if adjacent to any market tile
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.H and 0 <= nx < self.W and market_mask[ny, nx]:
                            queue.append((y, x, 0)) # Distance 0 at initial road node
                            self.dist_map[y, x] = 0
                            break

        # 2. Multi-source BFS along Road tiles
        while queue:
            cy, cx, d = queue.popleft()
            if d >= 15: # Stop propagation beyond score decay threshold
                continue
                
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.H and 0 <= nx < self.W and road_mask[ny, nx]:
                    if d + 1 < self.dist_map[ny, nx]:
                        self.dist_map[ny, nx] = d + 1
                        queue.append((ny, nx, d + 1))

        # 3. Calculate Score per House instance
        total_score = 0.0
        house_ids = set(np.unique(self.instance_grid[self.building_grid == 2]))
        house_ids.discard(0)
        
        for h_id in house_ids:
            # Find all road tiles adjacent to this 2x2 house
            h_coords = np.argwhere(self.instance_grid == h_id)
            min_dist = 99.0
            
            for hy, hx in h_coords:
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = hy + dy, hx + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W and road_mask[ny, nx]:
                        min_dist = min(min_dist, self.dist_map[ny, nx])
            
            # Apply decay formula
            if min_dist <= 10.0:
                total_score += 1.0
            elif 10.0 < min_dist <= 15.0:
                total_score += max(0.0, 1.0 - 0.2 * (min_dist - 10.0))
                
        return total_score