import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
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
            "market": 4,
            "pub" : 4
        }
        self.ACTION_BUILDINGS = ["road", "house", "market", "pub"]
        self.start_seed = start_seed
        self.base_score_change = 0
        self.BULIDING_SCORES = {
            "road": 0,
            "house": 0,
            "market": -0.5,
            "pub" : -0.5
        }
        self.house_adj_bonus_add = 0 # 
        self.house_adj_bonus_mult = 1.25 # 
        # Footprint sizes: (height, width)
        self.BUILDING_SIZES = {
            "road": (1, 1), # Road 
            "house": (1, 1), # House 
            "market": (2, 2),  # Market 
            "pub" : (2, 2)
        }
        self.longest_bulding = 5
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
        self.instances_dict = defaultdict(dict)
        
        # Road and house distance map 
        self.dist_maps = {  #only road tiles carry distance
            "market": np.full((self.H, self.W), 999, dtype=np.float32),
            "pub": np.full((self.H, self.W), 999, dtype=np.float32)
        }

        self.adjecent_houses_grid = np.zeros((self.H, self.W), dtype=np.int32)

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

        self.blueprint_mask = np.full((10, 10), None,  dtype=object) # holds info which actions each (empty) cell allows 
        # SEEDED START: Pre-place 1 Market and 1 adjacent Road in the center
        start_seed = (
            self.np_random.integers(0, 7)
            if self.start_seed == 0
            else self.start_seed
        )

        self._seeded_start(start_seed)
        
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

        for i, b_name in enumerate(["market", "pub"]): #market ranges, pub ranges
            road_only_dist_map = np.where(
            self.building_grid == 1,
            self.dist_maps[b_name],
            0.0
        )
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
                            obs[4 + i, y, x] = max(adjacent_values)

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
    def helper(self, y, x): 
        # input is and empty cell with road tile cell lower (y + 1 since (0, 0) is top left corner), it goes through all the building and for possbile places to build them, they are coded to action mask by their upper left corner, the most the can you got the left or right is is their lengh so that they touch this cell with bottom conrer. 
    
        DIRECTIONS = {
            "north": ((-1, 0), (0, 1)),
            "south": ((1, 0), (0, 1)),
            "east":  ((0, 1), (1, 0)),
            "west":  ((0, -1), (1, 0)),
        }     
        DIRECTION_NAMES = {
            (-1, 0): "north",
            (1, 0):  "south",
            (0, 1):  "east",
            (0, -1): "west",
        }

        road_id = self.BUIDLING_IDS_NAMES_BMAP.get_id("road")
        bh, bw = self.BUILDING_SIZES["road"]
        for ny, nx in self._get_orthogonal_neighbors(self.building_grid, y, x, bh, bw):
            if self.building_grid[ny, nx] == road_id:
                direction = DIRECTION_NAMES[(y - ny, x - nx)]
                break
        else:
            return self._get_action_mask()  # No adjacent road
        
        (fy, fx), (ly, lx) = DIRECTIONS[direction]

        mask = self._get_action_mask()
        L = self.longest_bulding
        heights = [0] * (L * 2 - 1) # stores info how far up can you go before hitting obstacle
        middle = L - 1

        for h in range(L):
            ny, nx = y + h * fy, x + h * fx

            if (0 <= ny < self.H and 0 <= nx < self.W
                and self.building_grid[ny, nx] == 0
                and self.land_mask[ny, nx] == 1):
                heights[middle] += 1
            else:
                break

        # Calculate available space on both lateral sides
        for i in range(1, L):
            for sign in [-1, 1]:
                offset = i * sign
                j = middle + offset

                # Cannot extend farther than the previous column
                max_height = heights[j - sign]

                for h in range(max_height):
                    ny = y + h * fy + offset * ly
                    nx = x + h * fx + offset * lx

                    if (0 <= ny < self.H and 0 <= nx < self.W
                        and self.building_grid[ny, nx] == 0
                        and self.land_mask[ny, nx] == 1):
                        heights[j] += 1
                    else:
                        break

        # Check each available building
        for action_b_id, b_name in enumerate(self.ACTION_BUILDINGS): # go through the avaible buildings
            bh, bw = self.BUILDING_SIZES[b_name]
            if direction in ("north", "south"):
            # The building dimensions NEVER rotate
                forward_len = bh
                lateral_len = bw
            else:
                forward_len = bw
                lateral_len = bh
            if heights[middle] < forward_len:
                continue

            l = r = 1

            while l < L and heights[middle - l] >= forward_len:
                l += 1

            while r < L and heights[middle + r] >= forward_len:
                r += 1

            # Valid starting offsets: the building must contain (y, x)
            # along the edge facing the road
            first = max(1 - l, 1 - lateral_len)
            last = min(0, r - lateral_len)

            if first > last:
                continue

            queue = deque([-1] * lateral_len, maxlen=lateral_len) #add dummy elememnets and so that you start popping real indexes after you go past where first building from the left ends (x - l + its wdith)
            
            for offset in range(1 - l, r):
                idx = -1

                if first <= offset <= last:
                    # Always calculate the NORTH-WEST corner
                    if direction == "north":
                        top_y, top_x = y - bh + 1, x + offset
                    elif direction == "south":
                        top_y, top_x = y, x + offset
                    elif direction == "east":
                        top_y, top_x = y + offset, x
                    else:  # west
                        top_y, top_x = y + offset, x - bw + 1

                    idx = (
                        action_b_id * (self.H * self.W)
                        + top_y * self.W
                        + top_x
                    )

                    mask[idx] = True

                queue.append(idx)

                actions = {a for a in queue if a != -1}

                if not actions:
                    continue

                # Add all actions covering cells in this lateral strip
                for h in range(forward_len):
                    ny = y + h * fy + offset * ly
                    nx = x + h * fx + offset * lx

                    if self.blueprint_mask[ny, nx] is None:
                        self.blueprint_mask[ny, nx] = set()

                    self.blueprint_mask[ny, nx].update(actions)

        return mask
                              
    def _update_action_mask(self):
        mask = self._get_action_mask()

        # Generate new valid building placements
        for y, x in self.allowed_buidling_builds:
            if (0 <= y < self.H and 0 <= x < self.W
                and self.building_grid[y, x] == 0
                and self.land_mask[y, x] == 1):
                self.helper(y, x)

        # Find all non-road placements covering extreme points
        blocked_actions = set()
        area = self.H * self.W

        for direction in self.extreme_points.values():
            for y, x in direction["points"]:
                if not (0 <= y < self.H and 0 <= x < self.W):
                    continue

                for idx in self.blueprint_mask[y, x] or ():
                    action_b_id = idx // area

                    if self.ACTION_BUILDINGS[action_b_id] != "road":
                        blocked_actions.add(idx)

        previous_blocked = getattr(self, "_blocked_by_extremes", set())

        # Restore placements that are no longer blocked by extremes
        for idx in previous_blocked - blocked_actions:
            position = idx % area
            top_y, top_x = divmod(position, self.W)

            # Only restore placements that still exist
            if (self.blueprint_mask[top_y, top_x] is not None
                and idx in self.blueprint_mask[top_y, top_x]):
                mask[idx] = True

        # Disable placements covering current extreme points
        for idx in blocked_actions:
            mask[idx] = False

        self._blocked_by_extremes = blocked_actions

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
    
    def _del_blueprints(self, y, x):

        
        blueprints = list(self.blueprint_mask[y, x] or ())
        if not blueprints:
            return
        
        mask = self._get_action_mask()

        for idx in blueprints:
            b_id = idx // (self.H * self.W)
            blueprint_building_name = self.ACTION_BUILDINGS[b_id]
            rem = idx % (self.H * self.W)
            blueprint_y, blueprint_x = rem // self.W, rem % self.W
            bh, bw = self.BUILDING_SIZES[blueprint_building_name]
            for ny in range(blueprint_y, blueprint_y + bh):
                for nx in range(blueprint_x, blueprint_x + bw):
                    self.blueprint_mask[ny, nx].discard(idx)
            mask[idx] = False

    def _place_tiles_on_grid(self, y, x, b_name):
        bh, bw = self.BUILDING_SIZES[b_name]
        b_id = self.BUIDLING_IDS_NAMES_BMAP.get_id(b_name)
        footprint = self.building_grid[y:y+bh, x:x+bw]
        land = self.land_mask[y:y+bh, x:x+bw]

        if footprint.shape != (bh, bw) or not np.all(footprint == 0) or not np.all(land == 1):
            raise RuntimeError("Taken / unavailable tile selected")

        for in_y in range(y, y + bh):
            for in_x in range(x, x + bw):
                self._del_blueprints(in_y, in_x)
                self.disallowed_building_build.add((in_y, in_x))

        self.building_grid[y:y+bh, x:x+bw] = b_id
        if b_name != "road": # roads dont need their id
            self.instance_grid[y:y+bh, x:x+bw] = self.next_instance_id
            self.instances_dict[self.next_instance_id] = {
                "building_name": b_name,
                "distances": {
                    "market": 999,
                    "pub": 999
                },
                "bounds": {
                    "top_left": (y, x),
                    "bottom_right": (y + bh, x + bw)
                },
                "neighbors": {
                    "house" : 0
                }
            }
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
                    s = str(int(self.instances_dict[self.instance_grid[y, x]]["distances"]["market"])) 
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
                    s = str(int(self.instances_dict[self.instance_grid[y, x]]["distances"]["pub"])) 
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
            self._building_agent(0, 0, "road")
        elif seed == 1:
            self._building_agent(self.H - 1, 0, "road")
        elif seed == 2:
            self._building_agent(0, self.W - 1, "road")
        elif seed == 3:
            self._building_agent(self.H - 1, self.W - 1, "road")
        elif seed == 4:
            self._building_agent(mid_y, mid_x, "road")
        elif seed == 5:
            y = self.np_random.integers(0, self.H)
            x = self.np_random.integers(0, self.W)
            self._building_agent(y, x, "road")
        elif seed == 6:
            for x in range(self.W):
                self._building_agent(self.H // 2, x, "road")
            for y in range(self.H // 2):
                if self.building_grid[y, self.W // 2] == 0: 
                    self._building_agent(y, self.W // 2, "road")
            self._building_agent(mid_y - 1, mid_x - 1, "market")
            self._building_agent(mid_y + 2, mid_x + 2, "pub")
        elif seed == 7:
            self._building_agent(1, 5, "road")

    def _count_adjecent_houses(self, y, x):
        house_id = self.BUIDLING_IDS_NAMES_BMAP.get_id("house")
        bh, bw = self.BUILDING_SIZES["house"]
        neighbors = self._get_orthogonal_neighbors(self.building_grid == house_id, y, x, bh, bw)
        return len(neighbors)
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
        model_score_change = 0
        for b_name in ["market", "pub"]:
            queue = deque()
            queue.append((start_y, start_x))
            # BFS over roads that have new shortest connection to market
            while queue:
                y, x = queue.popleft() # zrobic prawdziwy grid z tym które pole sąsiaduje z iloma bokami z innnym domem
                dist = self.dist_maps[b_name][y, x]
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]: 
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W :
                        cur_building_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
                        if cur_building_name == "road" and self.dist_maps[b_name][ny, nx] > dist + 1: # check if neighbour road are now closer to public buidling
                            queue.append((ny, nx))
                            self.dist_maps[b_name][ny, nx] = dist + 1
                        elif cur_building_name == "house" and self.instances_dict[self.instance_grid[ny, nx]]["distances"][b_name] > dist: # check if neighbour house is now closer to public buidling
                            old_dist = self.instances_dict[self.instance_grid[ny, nx]]["distances"][b_name]
                            new_dist = dist

                            old_reward = self._normalize_public_building_reward(old_dist, b_name)
                            new_reward = self._normalize_public_building_reward(new_dist, b_name)

                            house_neighbors = self.instances_dict[self.instance_grid[ny, nx]]["neighbors"]["house"]

                            reward_change = new_reward - old_reward
                            weighted_change = reward_change * self.house_adj_bonus_mult ** house_neighbors

                            # print(
                            #     f"{b_name} | House ({ny},{nx}) | "
                            #     f"dist: {old_dist} -> {new_dist} | "
                            #     f"reward: {old_reward} -> {new_reward} | "
                            #     f"neighbors: {house_neighbors} | "
                            #     f"weighted: {weighted_change}"
                            # )

                            model_score_change += weighted_change
                            self.total_game_score += score_change
                            self.instances_dict[self.instance_grid[ny, nx]]["distances"][b_name] = dist
        return model_score_change
    def _handle_house(self, y, x) -> float:
        model_score_change = 0
        building_cost = self.base_score_change + self.BULIDING_SCORES["house"]
        bh, bw = self.BUILDING_SIZES["house"]
        neighbors = self._get_orthogonal_neighbors((self.building_grid),y,x,bh,bw)

        adj_land_tiles = self._get_orthogonal_neighbors(self.land_mask, y, x, bh, bw)
        for ny, nx in adj_land_tiles:
            nb_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
            if nb_name == "house": # update rewards from adjecency for houses that arleady existed and got new neighbor, it doesnt impact game score
                neigh_instance = self.instances_dict[self.instance_grid[ny,nx]]
                normalized_reward_sum = 0
                for b_name in ["market", "pub"]:
                    normalized_reward_sum += self._normalize_public_building_reward(neigh_instance["distances"][b_name], b_name)
                prev_model_sum = building_cost + normalized_reward_sum * self.house_adj_bonus_mult ** neigh_instance["neighbors"]["house"] + self.house_adj_bonus_add * neigh_instance["neighbors"]["house"]
                neigh_instance["neighbors"]["house"] += 1
                new_model_sum = building_cost + normalized_reward_sum * self.house_adj_bonus_mult ** neigh_instance["neighbors"]["house"] + self.house_adj_bonus_add * neigh_instance["neighbors"]["house"]
                model_score_change += new_model_sum - prev_model_sum
                self.instances_dict[self.instance_grid[y,x]]["neighbors"]["house"] += 1

        for b_name in ["market", "pub"]:
            min_dist = 999

            for ny, nx in neighbors:
                nb_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
                if nb_name == "road":
                    min_dist = min(min_dist, self.dist_maps[b_name][ny, nx])
            self.instances_dict[self.instance_grid[y, x]]["distances"][b_name] = min_dist
            normalized_reward = self._normalize_public_building_reward(min_dist, b_name)
            model_score_change += building_cost + normalized_reward * self.house_adj_bonus_mult ** self.instances_dict[self.instance_grid[y, x]]["neighbors"]["house"]
            self.total_game_score += normalized_reward
        model_score_change += self.house_adj_bonus_add * self.instances_dict[self.instance_grid[y, x]]["neighbors"]["house"]
        return model_score_change      
    
    def _handle_road(self, y, x) -> float:
        model_score_change = self.base_score_change + self.BULIDING_SCORES["road"]
        min_dists = {
            "market": 999,
            "pub": 999
            }
        for b_name in ["market", "pub"]:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    nb_name = self.BUIDLING_IDS_NAMES_BMAP.get_name(self.building_grid[ny, nx])
                    if nb_name == "road": #is next to other road:
                        min_dists[b_name] = min(min_dists[b_name], self.dist_maps[b_name][ny, nx] + 1)
                    elif nb_name == b_name: #is next to market
                        min_dists[b_name] = 0
                    elif nb_name == "empty": #possible next road or building
                        self._update_extreme_points(ny,nx)
                        self.allowed_buidling_builds.add((ny,nx))
            self.dist_maps[b_name][y, x] = min_dists[b_name]
        # BFS over roads that have new shortest connection to market
        
        model_score_change += self._update_road_distances(y, x)

        return model_score_change

    def _calculate_public_building_effects(self, y, x, b_name):
        model_score_change = self.base_score_change + self.BULIDING_SCORES[b_name]
        bh, bw = self.BUILDING_SIZES[b_name]
        road_neighbours = self._get_orthogonal_neighbors(self.building_grid == self.BUIDLING_IDS_NAMES_BMAP.get_id("road"), y, x, bh, bw)
        for ny, nx in road_neighbours:
            self.dist_maps[b_name][ny, nx] = 0
            model_score_change += self._update_road_distances(ny, nx)
        return model_score_change
    
    def _handle_market(self, y, x) -> float:
        return self._calculate_public_building_effects(y, x, "market")
    
    def _handle_pub(self, y, x):
        return self._calculate_public_building_effects(y, x, "pub")