import pandas as pd
import numpy as np
import yaml
import dateutil

# This reads the CSV file
def get_df(filename):
    '''load the CSV file as a pandas DataFrame'''
    df = pd.read_csv(filename,comment='#')
    yaml_data = parse_header_yaml(filename)
    if 'created_at' in yaml_data:
        # new format (from fview2 0.20.25 or 0.20.26)

        # convert time
        t0_str = yaml_data['created_at']
        t0 = dateutil.parser.parse(t0_str)
        t0_stamp = t0.timestamp()
        time_microseconds = df['time_microseconds']
        timestamp = t0_stamp + time_microseconds*1e-6
        df['timestamp'] = timestamp

        # convert orientation
        df['slope'] = np.tan(df['orientation_radians_mod_pi'].values)
    return df

def calculate_dt_1(times):
    '''based on lowest intervals between timestamps, calculate inter-frame-interval'''
    dts = times[1:]-times[:-1]
    order_idx = dts.argsort()
    first_dts = []
    for idx in order_idx:
        this_dt = dts[idx]
        if this_dt == 0.0:
            continue
        first_dts.append(this_dt)
        if len(first_dts) >= 1000:
            break
    if len(first_dts) < 1:
        raise RuntimeError('cannot calculate dt')
    dt = np.median(first_dts)
    return dt

def calculate_frame(times, max_allowable_error_seconds=0.020):
    '''infer a reasonable value for an integer frame number based on timestamp'''
    dt = calculate_dt_1(times)
    t0 = times[0]
    times_relative = times-t0
    frame = times_relative/dt
    frame = np.round(frame).astype(np.uint32)
    time_predicted = frame * dt + t0
    time_error = abs(times - time_predicted)
    if np.max(time_error) > max_allowable_error_seconds:
        raise RuntimeError('error larger than %s msec calculating frames'%(max_allowable_error_seconds*1000.0))
    return frame

def parse_header_yaml(fname):
    with open(fname, mode='r') as fd:
        buf = fd.read(10000)
        comment_lines = []
        in_yaml_comment = False
        lines = buf.strip().split('\n')
        for idx,line in enumerate(lines):
            if line.startswith('# -- start of object detection yaml config --'):
                in_yaml_comment = True
                end_yaml_str = '# -- end of object detection yaml config --'
                continue
            elif line.startswith('# -- start of yaml config --'):
                in_yaml_comment = True
                end_yaml_str = '# -- end of yaml config --'
                continue
            if in_yaml_comment:
                if line.startswith(end_yaml_str):
                    break
                assert line[0:2] == '# '
                comment_lines.append(line[2:])
        yaml_buf = '\n'.join(comment_lines)
        yaml_data = yaml.safe_load(yaml_buf)
    return yaml_data

def parse_obj_dection_yaml(fname):
    yaml_data = parse_header_yaml(fname)
    if 'object_detection_cfg' in yaml_data:
        # new format (from fview2 0.20.25 or 0.20.26)
        yaml_data = yaml_data['object_detection_cfg']
    return yaml_data
