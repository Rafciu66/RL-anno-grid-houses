import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import random
from collections import defaultdict

class BidirectionalMap:
    def __init__(self, initial_data=None):
        self._id_to_name = {}
        self._name_to_id = {}
        
        if initial_data:
            # Accepts a dictionary ({id: name}) or a list of tuples ([(id, name), ...])
            items = initial_data.items() if isinstance(initial_data, dict) else initial_data
            for item_id, name in items:
                self.put(item_id, name)

    def put(self, item_id, name):
        # Clean up old mappings if an ID or Name is being reassigned
        if item_id in self._id_to_name:
            old_name = self._id_to_name[item_id]
            del self._name_to_id[old_name]
        if name in self._name_to_id:
            old_id = self._name_to_id[name]
            del self._id_to_name[old_id]

        self._id_to_name[item_id] = name
        self._name_to_id[name] = item_id

    def get_name(self, item_id):
        return self._id_to_name.get(item_id)

    def get_id(self, name):
        return self._name_to_id.get(name)

    def remove_by_id(self, item_id):
        name = self._id_to_name.pop(item_id, None)
        if name:
            self._name_to_id.pop(name, None)

    def remove_by_name(self, name):
        item_id = self._name_to_id.pop(name, None)
        if item_id is not None:
            self._id_to_name.pop(item_id, None)
    def __iter__(self):
        # Enables direct iteration: for item_id, name in mapping:
        return iter(self._id_to_name.items())

    def __len__(self):
        return len(self._id_to_name)
    
class IslandCityEnv(gym.Env):
    """
    Custom Gymnasium Environment for Island City Building with 5-channel observations
    and multi-tile Action Masking.
    """
    metadata = {"render_modes": []}

    def __init__(self, height=12, width=12, max_steps=100, start_seed = 0):
        super().__init__()
        self.H = height
        self.W = width
        self.max_steps = max_steps
        self.BUIDLING_IDS_NAMES_BMAP = BidirectionalMap({
            0 : "empty",
            1 : "road",
            2 : "house",
            3 : "market",
            4 : "pub"
        })
        self.PUBLIC_BUILDING_RANGES = {
            "market": 2,
            "pub" : 2
        }
        self.ACTION_BUILDINGS = ["road", "house", "market", "pub"]
        self.start_seed = start_seed
        self.base_score_change = 0
        self.BULIDING_SCORES = {
            "road": 0,
            "house": 0,
            "market": -0.25,
            "pub" : -0.25
        }
        self.house_adj_bonus_add = 0.25
        self.house_adj_bonus_mult = 1 # not working as inteded yet
        # Footprint sizes: (height, width)
        self.BUILDING_SIZES = {
            "road": (1, 1), # Road 
            "house": (1, 1), # House 
            "market": (1, 1),  # Market 
            "pub" : (1, 1)
        }
        self.BUIDLING_HANDLERS = {
            "road": self._handle_road,
            "house": self._handle_house,
            "market": self._handle_market,
            "pub" : self._handle_pub
        }
        # Action Space: 4 buildings * H * W
        self.n_buildings = 4
        self.action_space = spaces.Discrete(self.n_buildings * self.H * self.W)
        
        # Observation Space: (6, H, W)
        # Ch 0: Land Mask | Ch 1: Empty tiles | Ch 2: Roads | Ch 3: Houses | Ch 4: Market connectivity map | Ch 5: Pub connectivity map
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, self.H, self.W), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_model_score = 0.0
        self.total_game_score = 0.0
        # 1 = Land, 0 = Water
        self.land_mask = np.ones((self.H, self.W), dtype=np.float32)
        
        # Grid containing building IDs (0=empty, 1=road, 2=house, 3=market)
        self.building_grid = np.zeros((self.H, self.W), dtype=np.int8)
        
        # Unique IDs map to track multi-tile house/market instances
        self.instance_grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.next_instance_id = 1
        self.instance_distances_dict = defaultdict(dict)
        
        # Road and house distance map 
        self.dist_maps = {  #only road tiles carry distance
            "market": np.full((self.H, self.W), 999, dtype=np.float32),
            "pub": np.full((self.H, self.W), 999, dtype=np.float32)
        }

        self.action_mask = np.zeros(self.action_space.n, dtype=bool)
        # action mask for roads only
        self.allowed_buidling_builds = set()
        self.disallowed_building_build = set()
        self.updated_tiles = set()
        self.extreme_points = {
            "north": {"points": set(), "disabled": False},
            "south": {"points": set(), "disabled": False},
            "east":  {"points": set(), "disabled": False},
            "west":  {"points": set(), "disabled": False}
        }
        # SEEDED START: Pre-place 1 Market and 1 adjacent Road in the center
        self._seeded_start(random.randint(0,6) if self.start_seed == 0 else self.start_seed)
        
        #self._update_distances_and_score() 
        return self._get_obs(), {"action_mask": self.action_masks()}

    def _get_obs(self):
        """Constructs the 6-channel observation tensor (6, H, W)."""
        obs = np.zeros((6, self.H, self.W), dtype=np.float32)
        obs[0] = self.land_mask # ocean/mountains or land
        obs[1] = (self.building_grid == 0).astype(np.float32) # empty tiles
        obs[2] = (self.building_grid == 1).astype(np.float32) # road tiles
        obs[3] = (self.building_grid == 2).astype(np.float32) # house tiles
        
        # Normalize distance channel to [0, 1] relative to max scoring range (self.market_range)
        road_only_dist_map = np.where(
            self.building_grid == 1,
            self.dist_maps["market"],
            0.0
        )
        for i, b_name in enumerate(["market", "pub"]): #market ranges, pub ranges
            obs[4 + i] = np.where( # expected reward from placing house at each cell
                self.building_grid == self.BUIDLING_IDS_NAMES_BMAP.get_id("road"),
                self._normalize_public_building_reward(road_only_dist_map, b_name),
                0.0
            )
            # Propagate road distance values to adjacent empty cells
            original_obs = obs[4 + i].copy() 

            for y in range(self.H):
                for x in range(self.W):
                    if self.building_grid[y, x] == self.BUIDLING_IDS_NAMES_BMAP.get_id("empty"):
                        adjacent_values = [] 

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx

                            if 0 <= ny < self.H and 0 <= nx < self.W:
                                value = original_obs[ny, nx]

                                if value > 0:
                                    adjacent_values.append(value)

                        if adjacent_values:
                            obs[4, y, x] = max(adjacent_values)

        return obs
    def _building_agent(self, y, x, b_name) -> float: # returns change in score
        self._place_tiles_on_grid(y, x, b_name)
        handler = self.BUIDLING_HANDLERS.get(b_name)
        if handler:
            model_score_change = handler(y, x)
        else:
            model_score_change = 0
        self._update_action_mask()
        return model_score_change
    
    def action_masks(self) -> np.ndarray:
        self._update_action_mask()
        return self._get_action_mask()
    def _remove_extremes_from_set(self, yx_indexes):
        # 1. Collect all points from all directions into a single set
        all_extremes = {
            point
            for direction in self.extreme_points.values()
            for point in direction["points"]
        }

        # 2. Remove them from yx_indexes
        yx_indexes.difference_update(all_extremes)

    def _get_action_mask(self):
        return self.action_mask
    def _update_action_mask(self):
        mask = self._get_action_mask() # we dont clean it because we remove illegal actions and add new actions on the fly

        for action_b_id, b_name in enumerate(self.ACTION_BUILDINGS): # we need to use ACTION_BUILDINGS unitl action that makes cell empty is added
            bh, bw = self.BUILDING_SIZES[b_name]
            yx_indexes = self.allowed_buidling_builds
            if b_name != "road": # prevent roads that lead furthest from being blocked by skipping those indexes and keeping them False until they are outdated
                self._remove_extremes_from_set(yx_indexes) # this modifies set in place
            for y, x in yx_indexes:
                
                idx = action_b_id * (self.H * self.W) + y * self.W + x
                
                # 1. Boundary & Land check
                if y + bh > self.H or x + bw > self.W:
                    continue
                if not np.all(self.land_mask[y:y+bh, x:x+bw] == 1):
                    continue
                    
                footprint = self.building_grid[y:y+bh, x:x+bw]
                
                # Must be completely empty land to build
                if not np.all(footprint == 0):
                    continue
                
                
                if b_name == "road": # ROAD: must be placed always next to road - True for all allowed indexes
                    mask[idx] = True
                        
                        
                elif b_name == "house": # HOUSE: must be placed always next to road - True for all allowed indexes
                    mask[idx] = True
                        
                        
                elif b_name == "market": # MARKET: must be placed always next to road - True for all allowed indexes
                    mask[idx] = True

                elif b_name == "pub":
                    mask[idx] = True
                    

        for action_b_id, b_name in enumerate(self.ACTION_BUILDINGS): # switch cells to False that were blocked by placing buildings
            for y, x in self.disallowed_building_build:
                idx = action_b_id * (self.H * self.W) + y * self.W + x
                mask[idx] = False
        self.allowed_buidling_builds.clear() 
        self.disallowed_building_build.clear() 

    def _normalize_public_building_reward(self, distance, public_building):
      # Restrict (distance - building range) to values between 0 and building range
      b_range = self.PUBLIC_BUILDING_RANGES[public_building]
      clipped_dist = np.clip(
          distance - b_range, 0, b_range
      )
      return 1.0 - (clipped_dist / b_range)

    def step(self, action):
        self.current_step += 1
        
        # Decode action index -> (btype, y, x)
        b_id = action // (self.H * self.W)
        b_name = self.ACTION_BUILDINGS[b_id]
        rem = action % (self.H * self.W)
        y, x = rem // self.W, rem % self.W
        
        
        # Execute action
        reward = self._building_agent(y, x, b_name) 
            
        # Re-evaluate pathfinding & score
        self.total_model_score += reward

        mask = self.action_masks()
        terminated = not np.any(mask)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {"action_mask": mask}

    def _place_tiles_on_grid(self, y, x, b_name):
        bh, bw = self.BUILDING_SIZES[b_name]
        b_id = self.BUIDLING_IDS_NAMES_BMAP.get_id(b_name)

        for in_y in range(y, y + bh): #iterate thorugh tiles building will occupy
            for in_x in range(x, x+bw):
                self.disallowed_building_build.add((in_y, in_x))
        self.building_grid[y:y+bh, x:x+bw] = b_id
        if b_name != "road": # roads dont need their id
            self.instance_grid[y:y+bh, x:x+bw] = self.next_instance_id
            self.next_instance_id += 1

    def _get_orthogonal_neighbors(self, grid, y, x, bh, bw) -> list:
        """Returns (y, x) coordinates of all True cells adjacent to a rectangle."""
        neighbors = []

        # Top edge
        if y > 0:
            row = grid[y - 1, x:x + bw]
            for offset, val in enumerate(row):
                if val:
                    neighbors.append((y - 1, x + offset))

        # Bottom edge
        if y + bh < self.H:
            row = grid[y + bh, x:x + bw]
            for offset, val in enumerate(row):
                if val:
                    neighbors.append((y + bh, x + offset))

        # Left edge
        if x > 0:
            col = grid[y:y + bh, x - 1]
            for offset, val in enumerate(col):
                if val:
                    neighbors.append((y + offset, x - 1))

        # Right edge
        if x + bw < self.W:
            col = grid[y:y + bh, x + bw]
            for offset, val in enumerate(col):
                if val:
                    neighbors.append((y + offset, x + bw))

        return neighbors
    def render(self):
        """Renders the current city layout to the terminal using ASCII/Emojis."""
        symbols = {
            "empty": " 🟩",  # Empty Land
            "road": " ⬛",  # Road
            "house": " 🏠",  # House (2x2)
            "market": " 🏢",   # Market (3x3)
            "pub" : " 🍺"
        }
        
        print(f"\n--- Step: {self.current_step}/{self.max_steps} | Model score: {self.total_model_score:.2f}  | Game score: {self.total_game_score}---")
        #RED = "\033[31m"
        #GREEN = "\033[32m"
        #RESET = "\033[0m"
        for y in range(self.H):
            row_str = ""
            for x in range(self.W):
                b_id = self.building_grid[y, x]
                b_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(b_id)
                if b_name == "road":
                    s = str(int(self.dist_maps["market"][y, x]))
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                elif b_name == "house":
                    s = str(int(self.instance_distances_dict[self.instance_grid[y, x]]["market"])) 
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                else:
                    row_str += "##"
                row_str += "|"
            print(row_str)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for y in range(self.H):
            row_str = ""
            for x in range(self.W):
                b_id = self.building_grid[y, x]
                b_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(b_id)
                if b_name == "road":
                    s = str(int(self.dist_maps["pub"][y, x]))
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                elif b_name == "house":
                    s = str(int(self.instance_distances_dict[self.instance_grid[y, x]]["pub"])) 
                    row_str +=  s if len(s) >= 2 else str("0" + s)
                else:
                    row_str += "##"
                row_str += "|"
            print(row_str)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for y in range(self.H):
            row_str = ""
            for x in range(self.W):
                b_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[y, x])
                row_str += symbols.get(b_name, " ? ")
            print(row_str)
    def _seeded_start(self, seed):
        mid_y, mid_x = self.H // 2 - 1, self.W // 2 - 1
        if seed == 0:
            self._building_agent(0, 0, "road") # Road (1x1)
        if seed == 1:
            self._building_agent(self.H - 1, 0, "road") # Road (1x1)
        if seed == 2:
            self._building_agent(0, self.W - 1, "road") # Road (1x1)
        if seed == 3:
            self._building_agent(self.H - 1, self.W - 1, "road") # Road (1x1)
        if seed == 4:
            self._building_agent(mid_y, mid_x, "road") # Road (1x1)
        if seed == 5:
            self._building_agent(random.randint(0,self.H  - 1), random.randint(0,self.W  - 1), "road") # Road (1x1)
        if seed == 6:
            for x in range(self.W):
                self._building_agent(self.H // 2, x, "road")
            for y in range(self.H):
                self._building_agent(y, self.W // 2, "road")
            self._building_agent(mid_y, mid_x, "market") # Market (1x1)

    def _update_extreme_points(self, y, x):
        # 1. NORTH
        north = self.extreme_points["north"]
        if not north["disabled"]:
            if y == 0:
                north["points"].clear()
                north["disabled"] = True
            elif not north["points"] or y < next(iter(north["points"]))[0]:
                north["points"].clear()
                north["points"].add((y, x))
            elif y == next(iter(north["points"]))[0]:
                north["points"].add((y, x))

        # 2. SOUTH
        south = self.extreme_points["south"]
        if not south["disabled"]:
            if y == self.H - 1:
                south["points"].clear()
                south["disabled"] = True
            elif not south["points"] or y > next(iter(south["points"]))[0]:
                south["points"].clear()
                south["points"].add((y, x))
            elif y == next(iter(south["points"]))[0]:
                south["points"].add((y, x))

        # 3. EAST
        east = self.extreme_points["east"]
        if not east["disabled"]:
            if x == self.W - 1:
                east["points"].clear()
                east["disabled"] = True
            elif not east["points"] or x > next(iter(east["points"]))[1]:
                east["points"].clear()
                east["points"].add((y, x))
            elif x == next(iter(east["points"]))[1]:
                east["points"].add((y, x))

        # 4. WEST
        west = self.extreme_points["west"]
        if not west["disabled"]:
            if x == 0:
                west["points"].clear()
                west["disabled"] = True
            elif not west["points"] or x < next(iter(west["points"]))[1]:
                west["points"].clear()
                west["points"].add((y, x))
            elif x == next(iter(west["points"]))[1]:
                west["points"].add((y, x))
    def _update_road_distances(self, start_y, start_x) -> float:
        score_change = 0
        for b_name in ["market", "pub"]:
            queue = deque()
            queue.append((start_y, start_x))
            # BFS over roads that have new shortest connection to market
            while queue:
                y, x = queue.popleft()
                dist = self.dist_maps[b_name][y, x]
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]: 
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W :
                        cur_building_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
                        if cur_building_name == "road" and self.dist_maps[b_name][ny, nx] > dist + 1: # check if neighbour road are now closer to public buidling
                            queue.append((ny, nx))
                            self.dist_maps[b_name][ny, nx] = dist + 1
                        elif cur_building_name == "house" and self.instance_distances_dict[self.instance_grid[ny, nx]][b_name] > dist: # check if neighbour house is now closer to public buidling
                            score_change += self._normalize_public_building_reward(dist, b_name) - self._normalize_public_building_reward(self.instance_distances_dict[self.instance_grid[ny, nx]][b_name], b_name)
                            self.instance_distances_dict[self.instance_grid[ny, nx]][b_name] = dist
        return score_change
    def _handle_house(self, y, x) -> float:
        model_score_change = self.base_score_change + self.BULIDING_SCORES["house"]
        bh, bw = self.BUILDING_SIZES["house"]
        neighbors = self._get_orthogonal_neighbors((self.building_grid),y,x,bh,bw)
        min_dists = {
            "market": 999,
            "pub": 999
            }
        multiplicate = 1
        for ny, nx in neighbors:
            nb_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
            if nb_name == "road":
                min_dists["market"] = min(min_dists["market"], self.dist_maps["market"][ny, nx])
                min_dists["pub"] = min(min_dists["pub"], self.dist_maps["pub"][ny, nx])
            elif nb_name == "house":
                model_score_change+=self.house_adj_bonus_add
                multiplicate*=self.house_adj_bonus_mult
        self.instance_distances_dict[self.next_instance_id - 1]["market"] = min_dists["market"]
        self.instance_distances_dict[self.next_instance_id - 1]["pub"] = min_dists["pub"]
        normalized_reward = self._normalize_public_building_reward(min_dists["market"], "market")
        normalized_reward = self._normalize_public_building_reward(min_dists["pub"], "pub")
        model_score_change +=  normalized_reward* multiplicate
        self.total_game_score += normalized_reward

        return model_score_change      
    
    def _handle_road(self, y, x) -> float:
        model_score_change = self.base_score_change + self.BULIDING_SCORES["road"]
        min_dists = {
            "market": 999,
            "pub": 999
            }
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                nb_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
                if nb_name == "road": #is next to other road:
                    min_dists["market"] = min(min_dists["market"], self.dist_maps["market"][ny, nx] + 1)
                elif nb_name == "market": #is next to market
                    min_dists["market"] = 0
                elif nb_name == "empty": #possible next road or building
                    self._update_extreme_points(ny,nx)
                    self.allowed_buidling_builds.add((ny,nx))
        self.dist_maps["market"][y, x] = min_dists["market"]
        # BFS over roads that have new shortest connection to market
        score_change = self._update_road_distances(y, x)
        model_score_change+= score_change
        self.total_game_score+= score_change

        return model_score_change

    def _calculate_public_building_effects(self, y, x, b_name):
        model_score_change = self.base_score_change + self.BULIDING_SCORES[b_name]
        bh, bw = self.BUILDING_SIZES[b_name]
        empty_neighbours = self._get_orthogonal_neighbors(self.building_grid == self.BUIDLING_IDS_NAMES_BMAP.get_id("empty"), y, x, bh, bw)
        road_neighbours = self._get_orthogonal_neighbors(self.building_grid == self.BUIDLING_IDS_NAMES_BMAP.get_id("road"), y, x, bh, bw)
        for ny, nx in road_neighbours:
            self.dist_maps[b_name][ny, nx] = 0
            score_change = self._update_road_distances(ny, nx)
            model_score_change += score_change
            self.total_game_score += score_change
        return model_score_change
    
    def _handle_market(self, y, x) -> float:
        return self._calculate_public_building_effects(y, x, "market")
    
    def _handle_pub(self, y, x):
        return self._calculate_public_building_effects(y, x, "pub")