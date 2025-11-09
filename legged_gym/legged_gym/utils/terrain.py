import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        if self.type in ['competition']:
            self.border = int(cfg.border_size/self.cfg.horizontal_scale)
            self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) 
            self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) 
        else:
            self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
            self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
            if cfg.curriculum:
                self.curiculum()
            elif cfg.selected:
                self.selected_terrain()
            else:    
                self.randomized_terrain()   

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

        
        self.heightsamples = self.height_field_raw
        if self.type in ['trimesh','competition']:
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        '''
            坡度 (slope):difficulty 值越高，坡度越陡，地形就会更具挑战性。
            阶梯高度 (step_height):difficulty 值越高，阶梯的高度也会增大，增加训练的难度。
            障碍物高度 (discrete_obstacles_height):difficulty 值越高，障碍物的高度也会增加。
            间隔 (gap_size):difficulty 还会影响生成的“空隙”或“坑”的大小和深度，增加训练的难度。
        '''
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            '''
            np.unravel_index(k, (num_rows, num_cols)) 将 1D 索引 k 转换为 2D 索引 (i, j)，表示该地形在整张地图中的行列编号。
            示例: 假设 num_rows=3, num_cols=4，那么 num_sub_terrains=3×4=12，编号 k 依次是 0~11。
            np.unravel_index 的转换如下:
            k=0  → (0,0)   k=1  → (0,1)   k=2  → (0,2)   k=3  → (0,3)
            k=4  → (1,0)   k=5  → (1,1)   k=6  → (1,2)   k=7  → (1,3)
            k=8  → (2,0)   k=9  → (2,1)   k=10 → (2,2)   k=11 → (2,3)
            '''
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # np.random.uniform(0, 1) 生成一个 0~1 之间的浮点数，用于随机选择地形类型。
            choice = np.random.uniform(0, 1) 

            # 从 [0.5, 0.75, 0.9] 这三个值中随机选择一个,随机选择地形难度
            difficulty = np.random.choice([0.5, 0.75, 0.9])

            # 给每一个具体的网格生成地形。
            terrain = self.make_terrain(choice, difficulty)

            # 把生成的地形添加到地图中。
            self.add_terrain_to_map(terrain, i, j) 
        
    def curiculum(self):
        '''
            curiculum 方法生成基于课程的地形。难度随着行数增加而逐渐增大，
            choice 根据列数进行选择。同样，生成对应的地形并将其添加到地图中。
        '''
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 根据行列给出不同的难度和选择值，行列越大，难度越大，选择值越大
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                # 把难度和选择值给每一个具体的网格生成地形。
                terrain = self.make_terrain(choice, difficulty)

                # 把生成的地形添加到地图中。
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        '''
          给每一个具体的网格生成地形。
          make_terrain 方法根据传入的 choice 和 difficulty 生成地形。
          该方法首先创建一个 SubTerrain 对象，然后根据 choice 和 difficulty 生成不同类型的地形。

          make_terrain函数接受difficulty和choice两个参数，
          根据不同的choice生成不同类型的地形，而difficulty则决定地形的难度（such as 更高的台阶 更陡的坡） 
          具体地形的比例和类型，其实都是可以自己改的。
          比例在cfg类里面改，地形的话，isaacgym安装文件夹里面的terrain.py还有很多其他的地形可以直接用。
        '''
        # p.s.这里为什么要创建一个SubTerrain对象呢？因为要用isaacgym官方封装的库函数(terrain_utils.py)
        # 来生成地形。所以想调用这些函数就必须先创建一个 SubTerrain 对象。
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,             # 精细程度
                                horizontal_scale=self.cfg.horizontal_scale)         # 高度范围
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        '''
          slope:倾斜度，随着难度增加，倾斜度逐渐加大。
          step_height:步高，随着难度增加，步高增加。
          discrete_obstacles_height:障碍物的高度。
          stepping_stones_size:跳石的大小，随着难度增加，跳石的大小减小。
          stone_distance:跳石间距，难度为 0 时较小，其他难度较大。
          gap_size:间隙大小，随着难度增加而增大。
          pit_depth:坑深，随着难度增加，坑深增加。
        '''
        # choice 用来决定生成哪种类型的地形。它的值与 self.proportions 的数组进行比较，
        # 从而选择不同的地形生成方法。每个 choice 范围对应一种特定的地形类型。
        # 前置：假设LeggedRobotCfg.terrain.terrain_proportions = [0.1,   0.1,    0.35, 0.25, 0.2]
        # 各种地形类别占比：                                      平滑斜坡 崎岖斜坡 上楼梯 下楼梯 离散地形
        # 那么 执行该函数之前这个值会被传递并且改造成self.proportions = [0.1, 0.2, 0.55, 0.8, 1.0]（改成累加式）

        # 斜坡
        if choice < self.proportions[0]:
            # choice < self.proportions[0] / 2 则反转斜坡的方向（通过 slope *= -1）下坡
            # self.proportions[0]/2 < choice < self.proportions[0] 上坡
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # 崎岖斜坡
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # 增加随机噪声
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        # 上下樓梯
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        # 生成离散地形（石板地形）
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate syste
        # start_x, end_x：通过 i（行号）计算地形在 x 方向的起始和结束位置，单位是像素。
        # self.length_per_env_pixels 代表每个网格的宽度，self.border 是地图的边界偏移量
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 将生成的地形高度数据添加到地图
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        # env_origin_x 和 env_origin_y：计算环境的原点位置（中心点），
        # 分别基于行 i 和列 j 的位置。self.env_length 和 self.env_width 分别表示每个环境的长度和宽度。
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # x1, x2, y1, y2：这些是通过环境的大小、水平缩放比例来确定的索引，
        # 表示环境中心区域的范围。这些索引用于在 terrain.height_field_raw 中查找环境的最大高度。
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        
        # 通过在 terrain.height_field_raw 中查找环境中心区域的最大高度来计算该环境的 z 坐标（高度），
        # 然后将其乘以 vertical_scale 来进行缩放。
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        # 最后，将计算出的环境原点的坐标 [env_origin_x, env_origin_y, env_origin_z] 
        # 存储到 self.env_origins 数组中，表示该地形在整个地图上的位置。
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def obstest_terrain(self):
        # 这个方法的输入参数很重要，我们需要生成地形的长度和宽度，这决定了返回的height_field_raw数组大小。
        def new_sub_terrain(length = self.terrain_length, width = self.terrain_width): 
            '''
              定义 new_sub_terrain 方法
              new_sub_terrain 是一个辅助函数，创建一个新的 SubTerrain 对象
            '''
           
            return SubTerrain(terrain_name     = "terrain",
                              width            = width,
                              length           = length,
                              vertical_scale   = self.cfg.vertical_scale,
                              horizontal_scale = self.cfg.horizontal_scale)
        '''
        列col/length
        ^
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |---------------------------------------------------->行row/width
        
        按照不同列生成不同的地形，那就需要在每一组地形块的所有行生成不同的地形
        length_per_env_pixels:每一个地形块的行像素数
        width_per_env_pixels :每一个地形块的列像素数
        tot_rows:总行像素数
        tot_cols:总列像素数
        '''
        # 平坦地形
        self.height_field_raw[0*self.length_per_env_pixels:1*self.length_per_env_pixels,:] = (
            sloped_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), slope = 0.0).height_field_raw
        )
        
        # 上坡地形
        self.height_field_raw[1*self.length_per_env_pixels:2*self.length_per_env_pixels,:] = (
            pyramid_sloped_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), slope = -0.3).height_field_raw
        )
        
        # 随机地形
        self.height_field_raw[2*self.length_per_env_pixels:3*self.length_per_env_pixels,:] = (
             random_uniform_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                    min_height=-0.15, max_height=0.15, 
                                    step=0.2, downsampled_scale=0.5).height_field_raw
        )
        
        # 带障碍物的地形
        self.height_field_raw[3*self.length_per_env_pixels:4*self.length_per_env_pixels,:] = (
             discrete_obstacles_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                        max_height=0.15, min_size=1., max_size=5., num_rects=20).height_field_raw
        )
        
        # 波浪地形
        self.height_field_raw[4*self.length_per_env_pixels:5*self.length_per_env_pixels,:] = (
             wave_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                          num_waves=2., amplitude=1.).height_field_raw
        )
        
        # 楼梯地形
        self.height_field_raw[5*self.length_per_env_pixels:6*self.length_per_env_pixels,:] = (
             stairs_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                            step_width=0.75, step_height=0.25).height_field_raw
        )
        
        # 楼梯地形
        self.height_field_raw[6*self.length_per_env_pixels:7*self.length_per_env_pixels,:] = (
             stairs_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                            step_width=0.75, step_height=0.25).height_field_raw
        )
                
        # 跳石地形
        self.height_field_raw[7*self.length_per_env_pixels:8*self.length_per_env_pixels,:] = (
             stepping_stones_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                     stone_size=1.,stone_distance=0.25, 
                                     max_height=0.2, platform_size=0.).height_field_raw
        )