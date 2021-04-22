import pickle

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mplclrs

#from flydance.configuration.config_utils import read_, write_config_yaml

def get_xy_cols_data(df, preferred=None):
    if preferred is not None:
        xpref = 'x_{}'.format(preferred)
        ypref = 'y_{}'.format(preferred)
        if xpref in df.columns:
            return xpref, ypref
    if 'x_px' in df.columns:
        return 'x_px', 'y_px'
    if 'x' in df.columns:
        return 'x', 'y'
    raise NotImplementedError('unknown dataframe format')

def plot_trajectory(xs, ys, ax, colorful=True, markersize=4, scatter=True, tmax=None,
                    cmap=None, colors=None, **kwargs):
    ax.set(aspect='equal')
    rasterized = kwargs.get('rasterized', True)  # default value: True
    kwargs['rasterized'] = rasterized  # assign if it was not specified
    # print('raster:', kwargs['rasterized'])
    # print('plot_traj. colorful: {}, scatter: {}'.format(colorful, scatter))
    if colorful:
        # print('plotting colorful traj')
        if colors is None:
            colors = np.arange(xs.shape[0]) / (xs.shape[0] - 1)
            if tmax is not None:
                colors *= tmax
        sc = ax.scatter(xs, ys, marker='.', s=markersize, c=colors, lw=0, zorder=3, cmap=cmap,
                        label='_no_legend_', **kwargs)
        return sc
    if scatter:
        # print('plotting scatter traj')
        ax.scatter(xs, ys, marker='.', s=markersize, lw=0, zorder=3, **kwargs)
        return None
    # print('plotting line traj')
    ax.plot(xs, ys)
    return None


def circular_object_pathces(center, radius, plot_kwargs):
    my_patches = []
    plot_kwargs2 = plot_kwargs.copy()
    if 'zorder' not in plot_kwargs2:
        plot_kwargs2['zorder'] = 5

    plot_kwargs2['color'] = 'none'
    if plot_kwargs['color'] != 'none':
        # print('colored')
        loc_filling = plt.Circle(center, radius=radius, **plot_kwargs)
        my_patches.append(loc_filling)

    loc = plt.Circle(center, radius=radius, **plot_kwargs2)
    my_patches.append(loc)
    return my_patches


def circular_object_plot(ax, center, radius, plot_kwargs, indicate_center):
    my_patches = []
    plot_kwargs2 = plot_kwargs.copy()
    if 'zorder' not in plot_kwargs2:
        plot_kwargs2['zorder'] = 5

    plot_kwargs2['color'] = 'none'
    if plot_kwargs['color'] != 'none':
        # print('colored')
        loc_filling = plt.Circle(center, radius=radius, **plot_kwargs)
        ax.add_artist(loc_filling)
        my_patches.append(loc_filling)

    loc = plt.Circle(center, radius=radius, **plot_kwargs2)
    my_patches.append(loc)
    ax.add_artist(loc)
    if indicate_center:
        ax.scatter(center[0], center[1], marker='+', color=plot_kwargs['ec'])
    return my_patches


class WalkingFlyArena:
    def __init__(self, center_x, center_y, radius, radius_cm=None):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.objects = {}

        self.cmleft = -radius
        self.cmright = radius
        self.px_to_cm_ratio = 1
        self.radius_cm = radius_cm
        self.set_cm_radius(radius_cm)

    def set_cm_radius(self, cm_radius):
        if cm_radius is None:
            return 1
        self.radius_cm = cm_radius
        self.cmleft = -cm_radius
        self.cmright = cm_radius
        self.px_to_cm_ratio = self.radius / self.radius_cm
        return 0

    def plot(self, ax=None, with_objects=True, cm_ticks=False, with_centers=False, margin=0.05, axes_visible=True, lw=1):
        """
        :param lw: line width for arena border and objects
        :param ax: axis to draw on
        :param with_objects: draw objects or not (such as reward etc)
        :param cm_ticks: show ticks for cm scale
        :param with_centers: indicate centers for circular objects
        :param margin: margin (between ax border and circle)
        :param axes_visible: show axes
        :return: list of patches (circular objects). Useful for animations
        """
        x_min = self.center_x - self.radius
        x_max = self.center_x + self.radius
        y_min = self.center_y - self.radius
        y_max = self.center_y + self.radius
        margin = self.radius * margin
        # if not axes_visible:
        #     margin = 0

        if ax is None:
            fig, ax = plt.subplots()
        arena_border = plt.Circle([self.center_x, self.center_y], radius=self.radius, color='none', ec='black', lw=lw)
        # food_circle = plt.Circle([center_x+200,center_y], radius= 50, ec='red', color = 'None')

        patches = [arena_border]
        ax.add_artist(arena_border)
        # ax.add_artist(food_circle)
        ax.set(aspect='equal', xlim=[x_min - margin, x_max + margin], ylim=[y_min - margin, y_max + margin])

        if with_objects:
            for name in self.objects.keys():
                obj_patches = self.plot_object(name, ax, indicate_center=with_centers, lw=lw)
                patches += obj_patches
        if not axes_visible:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # ax.axis('off')
            # ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            return patches

        if cm_ticks:
            ax.set_xticks([self.center_x - self.radius, self.center_x, self.center_x + self.radius])
            # ax.set_xticklabels(['-10', '0', '10'])
            ax.set_xticklabels(['{:.0f}'.format(-self.radius_cm), '0', '{:.0f}'.format(self.radius_cm)])
            # ax.set_yticks([self.center_y - self.radius, self.center_y, self.center_y + self.radius])
            # ax.set_yticklabels(['-10', '0', '10'])
            ax.set_yticks([self.center_y - self.radius, self.center_y, self.center_y + self.radius])
            ax.set_yticklabels(['{:.0f}'.format(-self.radius_cm), '0', '{:.0f}'.format(self.radius_cm)])
        return patches

    def contains_point(self, x, y):
        """
        :param x: x-coordinate of the point
        :param y:
        :return: bool: true if point is inside the arena (not including border)
        """
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 < self.radius ** 2

    def plot_object(self, name, ax, indicate_center=True, lw=None):
        obj_patches = []

        obj = self.objects[name]
        if not obj.get('visible', True):
            # print(name, 'invisible')
            return []

        if obj['type'] == 'circle':
            plot_kwargs = obj.get('plot_kwargs', {'ec': obj['color'], 'color': 'none', 'linestyle': '-', 'zorder': 5})
            if lw is not None:
                if 'lw' not in plot_kwargs and 'linewidth' not in plot_kwargs:
                    plot_kwargs['lw'] = lw  # do not overwrite if exists
            obj_patches = circular_object_plot(ax=ax, center=(obj['x'], obj['y']), radius=obj['radius'],
                                               plot_kwargs=plot_kwargs, indicate_center=indicate_center)
            # lnst = obj.get('linestyle', '-')
        elif obj['type'] == 'xybins':
            for x in obj['xbins']:
                vline = ax.axvline(x)
                obj_patches.append(vline)
            for y in obj['ybins']:
                hline = ax.axhline(y)
                obj_patches.append(hline)
        else:
            raise NotImplementedError('Unrecognised type: {}'.format(obj.type))
        return obj_patches

    def get_artists(self, object_name):
        c = self.objects[object_name]
        if c['type'] != 'circle':
            raise NotImplementedError(c['type'])
        plot_kwargs = c.get('plot_kwargs', {'ec': c['color'], 'color': 'none', 'linestyle': '-', 'zorder': 5})
        obj_patches = circular_object_pathces(center=(c['x'], c['y']), radius=c['radius'],
                                              plot_kwargs=plot_kwargs)
        return obj_patches

    def add_circular_location(self, name, x, y, radius, color):
        self.objects[name] = {'type': 'circle',
                              'x': x, 'y': y, 'radius': radius,
                              'color': color
                              }

    def get_circular_location_data(self, name):
        c = self.objects[name]
        if c['type'] != 'circle':
            raise Exception('wrong type', c['type'])
        return c['x'], c['y'], c['radius']

    def get_opposite_coords(self, x, y):
        return self.center_x - (x - self.center_x), self.center_y - (y - self.center_y)

    def add_opposite_circ_object(self, existing_object_name, new_object_name, new_color):
        if existing_object_name not in self.objects:
            print('Error: There is no object {} in the arena'.format(existing_object_name))
            return
        source = self.objects[existing_object_name]
        x, y = self.get_opposite_coords(source['x'], source['y'])
        self.add_circular_location(new_object_name, x, y, source['radius'], new_color)

    def set_reward_location(self, x, y, r=None):
        if 'reward' in self.objects:
            self.objects['reward']['x'] = x
            self.objects['reward']['y'] = y
            if r:
                self.objects['reward']['radius'] = r
            if 'unrewarded' in self.objects:
                self.objects['unrewarded']['x'], self.objects['unrewarded']['y'] = self.get_opposite_coords(x, y)
                if r:
                    self.objects['unrewarded']['radius'] = r
        else:
            assert r is not None
            self.add_circular_location('reward', x, y, r, 'red')

    def plot_trajectory_df_speed_color_code(self, df, ax, markersize=4, logcolor=False):
        xs = df.x_px
        ys = df.y_px
        tcol = 't'
        if tcol not in df.columns:
            tcol = 'timestamp'
        # colors = np.arange(xs.shape[0]) / (xs.shape[0] - 1)

        dt = df[tcol].diff()
        dx = df.x_px.diff()
        dy = df.y_px.diff()
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        if logcolor:
            speed = np.log(speed)

        colors = speed / speed.max()
        # print(colors)

        sc = ax.scatter(xs, ys, marker='.', s=markersize, c=colors, lw=0)
        # self.plot(ax)
        return sc

    def has_object(self, name):
        return name in self.objects

    def xy_binning(self, nbins_x, nbins_y=None, visible=False):
        if nbins_x == 0:
            return
        if nbins_y is None:
            nbins_y = nbins_x
        x0 = self.center_x - self.radius
        x1 = self.center_x + self.radius
        y0 = self.center_y - self.radius
        y1 = self.center_y + self.radius

        xbins = np.linspace(x0, x1, nbins_x + 1)
        ybins = np.linspace(y0, y1, nbins_y + 1)

        xbin_centers = xbins[:-1] + np.diff(xbins) / 2
        ybin_centers = ybins[:-1] + np.diff(ybins) / 2

        self.objects['bins'] = {'type': 'xybins',
                                'xbins': xbins, 'ybins': ybins,
                                'xbin_centers': xbin_centers, 'ybin_centers': ybin_centers,
                                'visible': visible
                                }

    def get_nbins(self, axis=0):
        if 'bins' not in self.objects.keys():
            return 0
        if axis == 0:
            return len(self.objects['bins']['xbin_centers'])
        elif axis == 1:
            return len(self.objects['bins']['xbin_centers'])
        raise Exception('axis should be 0 or 1 for x or y bins respectively')

    def save_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def set_object_visibility(self, obj_name, visible):
        self.objects[obj_name]['visible'] = visible

#     def set_rz_from_led_trigger_config_yaml(self, config_yaml, name='reward', color='red'):
#         cfg = read_config_yaml(config_yaml)
#         self.set_rz_from_led_trigger_config_dict(cfg, name, color)

    def set_rz_from_led_trigger_config_dict(self, config_dict, name='reward', color='red'):
        # todo: 2-stage implementation, now 1-stage triggering only
        
        circle = config_dict["led_on_shape_pixels"]["Circle"]
        x = circle["center_x"]
        y = circle["center_y"]
        r = circle["radius"]
        self.add_circular_location(name=name, x=x, y=y, radius=r, color=color)

#     def save_object_to_yaml(self, yaml_filename, obj_name="reward"):
#         circle_dict = dict(center_x=self.objects[obj_name]["x"],
#                            center_y=self.objects[obj_name]["y"],
#                            radius=self.objects[obj_name]["radius"])
#         config_dict = {"led_on_shape_pixels": {"Circle": circle_dict}}
#         write_config_yaml(config_dict, yaml_filename)


def create_arena_from_config_dict(config):
    # used
    if 'pickle_file' in config:
        with open(config['pickle_file'], 'rb') as f:
            arena = pickle.load(f)
        return arena

    # deprecated
    x0 = config['center_x']
    y0 = config['center_y']
    r = config['radius_px']
    r_cm = config.get('radius_cm', None)
    px_to_cm_ratio = r / r_cm

    arena = WalkingFlyArena(x0, y0, r, r_cm)
    for loc_name in ['reward', 'reward_initiation']:
        if loc_name in config:
            loc_config = config[loc_name]
            rx = loc_config['dx'] + x0
            ry = loc_config['dy'] + y0
            arena.add_circular_location(loc_name, rx, ry, loc_config['radius'], loc_config['color'])
    return arena


def create_arena_from_yaml_data(yaml_data, locations_config=None):
    """
    :param yaml_data: arena config from object detection yaml
    :param locations_config: arena objects config stored
    separately -- x,y,r of arena are ignored, only object data is used
    :return: WalkingFlyArena
    """
    config = yaml_data['valid_region']['Circle']

    x0 = config['center_x']
    y0 = config['center_y']
    r = config['radius']
    arena = WalkingFlyArena(x0, y0, r)
    if locations_config is not None:
        for loc_name in ['reward', 'reward_initiation']:
            if loc_name in locations_config:
                loc_config = locations_config[loc_name]
                if 'dx' in loc_config:
                    rx = loc_config['dx'] + x0
                    ry = loc_config['dy'] + y0
                elif 'x' in loc_config:
                    rx = loc_config['x']
                    ry = loc_config['y']
                else:
                    raise Exception('Could not parse location config: {}'.format(loc_config))
                arena.add_circular_location(loc_name, rx, ry, loc_config['radius'], loc_config['color'])

        dimensions = locations_config['dimensions']
        px_to_cm_ratio = dimensions['px'] / dimensions['cm']
        print('r cm:', arena.radius / px_to_cm_ratio)
        arena.set_cm_radius(arena.radius / px_to_cm_ratio)

    return arena


def my_trajectory_colorbar(figure, ax, colors, val_min, val_max, intsecs=True):
    cbar = figure.colorbar(colors, ax=ax, ticks=[0, 1])
    if intsecs:
        cbar.ax.set_yticklabels(['{:.2f}'.format(val_min), '{:.2f}'.format(val_max)])
    else:
        cbar.ax.set_yticklabels(['{:.2f}'.format(val_min), '{:.2f}'.format(val_max)])


def get_arena_histogram_from_df(df, arena):
    """
    :param df:
    :param arena:
    :return: h, xe, ye: value coun
    """
    xbins = arena.objects['bins']['xbins']
    ybins = arena.objects['bins']['ybins']
    xcol, ycol = get_xy_cols_data(df)
    h, xe, ye = np.histogram2d(df[xcol], df[ycol], bins=(xbins, ybins))
    return h, xe, ye


def plot_arena_histogram(h, xe, ye, ax, arena, logscale=False, labeled_cbar=True, frame_visible=False,
                         vmin=None, vmax=None, cmap=None):
    h = h.T
    ntotal = h.sum()
    h_norm = h / ntotal
    h_norm[h_norm == 0] = np.nan
    h_norm_percent = h_norm * 100

    X, Y = np.meshgrid(xe, ye)
    norm = None
    if logscale:
        norm = mplclrs.LogNorm()
    im = ax.pcolormesh(X, Y, h_norm_percent, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0, rasterized=True)
    im.set_edgecolor('face')

    arena.plot(ax, cm_ticks=True, axes_visible=frame_visible, with_centers=False)
    if labeled_cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('residence frequency, %', rotation=270, labelpad=10)
    return im


def arena_hist2d(data_x, data_y, ax, arena, logscale=False, labeled_cbar=True, frame_visible=False,
                 vmin=None, vmax=None, cmap=None):
    xbins = arena.objects['bins']['xbins']
    ybins = arena.objects['bins']['ybins']
    h, xe, ye = np.histogram2d(data_x, data_y, bins=(xbins, ybins))
    return plot_arena_histogram(h, xe, ye, ax, arena, logscale, labeled_cbar, frame_visible, vmin, vmax, cmap)



def my_hist2d(data_x, data_y, ax, xbins, ybins, logscale=False, labeled_cbar=True, axes_visible=False,
              vmin=None, vmax=None, show_cbar=False, cmap=None):
    h, xe, ye = np.histogram2d(data_x, data_y, bins=(xbins, ybins))
    h = h.T
    h_norm = h / data_x.shape[0]
    h_norm[h_norm == 0] = np.nan
    h_norm_percent = h_norm * 100

    X, Y = np.meshgrid(xe, ye)
    norm = None
    if logscale:
        norm = mplclrs.LogNorm()
    im = ax.pcolormesh(X, Y, h_norm_percent, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax)
        if labeled_cbar:
            cbar.set_label('residence frequency, %', rotation=270, labelpad=10)

    if not axes_visible:
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    return im



def generate_cm_arena(arena, cm_radius):
    cm_arena = WalkingFlyArena(0, 0, cm_radius)
    kpxcm = cm_radius / arena.radius
    for obj_name, obj_sett in arena.objects.items():
        print(obj_name, obj_sett)
        if obj_sett['type'] == 'circle':
            rcm = kpxcm * obj_sett['radius']
            xcm = kpxcm * (obj_sett['x'] - arena.center_x)
            ycm = kpxcm * (obj_sett['y'] - arena.center_y)

            cm_arena.add_circular_location(obj_name, xcm, ycm, rcm, obj_sett['color'])
    return cm_arena


def load_arena_pickle(fname) -> WalkingFlyArena:
    with open(fname, 'rb') as f:
        arena = pickle.load(f)
        return arena
    