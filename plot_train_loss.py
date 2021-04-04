import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import re
import numpy as np
 
def get_log_parsing_script():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    return dirname + '/parse_log.sh'
 
def get_log_file_suffix():
    return '.log'
 
def get_chart_type_description_separator():
    return '  vs. '
 
def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields
 
def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' loss':2,
                              train_key + ' learning rate':3},
                   test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':2,
                             test_key + ' loss':3}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields
 
def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in range(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in range(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types
 
def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description
 
def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type
 
def get_data_file(chart_type, path_to_log):
    return (os.path.basename(path_to_log) + '.' +
            get_data_file_type(chart_type).lower())
 
def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field
 
def get_field_indices(x_axis_field, y_axis_field):
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]
 
def load_data(data_file, field_idx0, field_idx1):
    field_idx0=1
    field_idx1=9
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] == 'T':
                fields = line.split()
                a=fields[field_idx0].strip()
                epo=re.findall(r'\[(.+?)\]', a)[0]
                iters=re.findall(r'\[(.+?)\]', a)[1].split('/')[0]
                iters=int(epo)*10968+int(iters)
                data[0].append(float(iters))
                data[1].append(float(fields[field_idx1].strip()))
    return data
 
def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.keys())
    idx = random.randint(0, num - 1)
    return list(markers.keys())[idx]
 
def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label
 
def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc
 
def smooth(a,WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


def plot_chart(chart_type, path_to_png, path_to_log_list):
    for path_to_log in path_to_log_list:
        os.system('%s %s' % (get_log_parsing_script(), path_to_log))
        # data_file = get_data_file(chart_type, path_to_log)
        data_file = path_to_log_list[0]
        x_axis_field, y_axis_field = get_field_descriptions(chart_type)
        # x, y = get_field_indices(x_axis_field, y_axis_field)
        x=0
        y=0
        data = load_data(data_file, x, y)
        data_x=np.array(data[0])
        data_x=smooth(data_x,51)
        data_x=data_x[::20]
        data_y=np.array(data[1])
        data_y=smooth(data_y,51)
        data_y=data_y[::20]

        ## TODO: more systematic color cycle for lines
        color = [random.random(), random.random(), random.random()]
        label = get_data_label(path_to_log)
        linewidth = 0.75
        ## If there too many datapoints, do not use marker.
        use_marker = False
        # use_marker = True
        if not use_marker:
            plt.plot(data_x, data_y, label = label, color = color,
                     linewidth = linewidth)
        else:
            marker = random_marker()
            plt.plot(data[0], data[1], label = label, color = color,
                     marker = marker, linewidth = linewidth)
    legend_loc = get_legend_loc(chart_type)
    plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    plt.title(get_chart_type_description(chart_type))
    plt.xlabel(x_axis_field)
    plt.ylabel(y_axis_field)
    plt.savefig(path_to_png)
    plt.show()
 
def print_help():
    print ("""This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
 field index in create_field_index before designing your own plots.
Usage:
    ./plot_training_log.py chart_type[0-%s] /where/to/save.png /path/to/first.log ...
Notes:
    1. Supporting multiple logs.
    2. Log file name must end with the lower-cased "%s".
Supported chart types:""" , (len(get_supported_chart_types()) - 1,
                             get_log_file_suffix()))
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in range(num):
        print('    %d: %s',(i, supported_chart_types[i]))
    # sys.exit()
 
def is_valid_chart_type(chart_type):
    return chart_type >= 0 and chart_type < len(get_supported_chart_types())

def plot_chart_acc(chart_type, path_to_png, path_to_log_list):
    for path_to_log in path_to_log_list:
        os.system('%s %s' % (get_log_parsing_script(), path_to_log))
        # data_file = get_data_file(chart_type, path_to_log)
        data_file = path_to_log_list[0]
        x_axis_field, y_axis_field = get_field_descriptions(chart_type)
        # x, y = get_field_indices(x_axis_field, y_axis_field)
        x=0
        y=0
        # data = load_data(data_file, x, y)
        data=[]
        data.append(range(0,28))
        # train
        data.append([0.6762,0.8033,0.8175,0.8254,0.8290,0.8311,0.8324,0.8328,0.8341,0.8339,
                0.8345,0.8349,0.8352,0.8356,0.8362,0.8362,0.8370,0.8368,0.8375,0.8377,
                0.8384,0.8383,0.8383,0.8385,0.8396,0.8399,0.8390,0.8404])
        data.append([0.3155,0.5619,0.5642,0.5952,0.6416,0.6560,0.6403,0.6735,0.6768,0.6726,
                0.6811,0.6775,0.6820,0.6797,0.6841,0.6728,0.6910,0.6843,0.6850,0.6933,
                0.6915,0.6969,0.6830,0.6929,0.6837,0.6859,0.6921,0.6954])
        # test
        data.append([0.7443,0.7434,0.8101,0.8084,0.7896,0.8080,0.7880,0.8179,0.7849,0.7956,
                0.7934,0.8015,0.7985,0.7603,0.8176,0.7870,0.7945,0.7616,0.7845,0.8209,
                0.7673,0.8033,0.7841,0.8023,0.7899,0.7961,0.7985,0.8156])
        data.append([0.3894,0.4544,0.5555,0.6132,0.6109,0.5719,0.6024,0.6603,0.6112,0.6581,
                0.6381,0.6077,0.6350,0.6162,0.6849,0.6525,0.6476,0.6237,0.6081,0.6269,
                0.6052,0.6482,0.6554,0.6423,0.6452,0.6322,0.6221,0.6710])
        ## TODO: more systematic color cycle for lines
        color = [random.random(), random.random(), random.random()]
        label = get_data_label(path_to_log)
        linewidth = 0.75
        ## If there too many datapoints, do not use marker.
##        use_marker = False
        use_marker = True
        if not use_marker:
            plt.plot(data[0], data[1], label = label, color = color,
                     linewidth = linewidth)
        else:
            marker = random_marker()
            plt.plot(data[0], data[1], label = 'train_OP', color = color,
                     marker = marker, linewidth = linewidth)
            color = [random.random(), random.random(), random.random()]
            plt.plot(data[0], data[2], label = 'train_CP', color = color,
                     marker = marker, linewidth = linewidth)
            color = [random.random(), random.random(), random.random()]
            plt.plot(data[0], data[3], label = 'test_OP', color = color,
                     marker = marker, linewidth = linewidth)
            color = [random.random(), random.random(), random.random()]
            plt.plot(data[0], data[4], label = 'test_CP', color = color,
                     marker = marker, linewidth = linewidth)

    legend_loc = get_legend_loc(chart_type)
    plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    plt.title('Train and Test precision vs Epoch\n average per-class precision (CP)\n average overall precision (OP)')
    plt.xlabel('Epoch')
    # plt.ylabel()
    plt.savefig(path_to_png)
    plt.show()
 
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_help()
    else:
        chart_type = int(sys.argv[1])
        if not is_valid_chart_type(chart_type):
            print ('%s is not a valid chart type.', chart_type)
            print_help()
        path_to_png = sys.argv[2]
        if not path_to_png.endswith('.png'):
            print ('Path must ends with png', path_to_png)
            sys.exit()
        path_to_logs = sys.argv[3:]
        for path_to_log in path_to_logs:
            if not os.path.exists(path_to_log):
                print ('Path does not exist: %s',path_to_log)
                sys.exit()
            if not path_to_log.endswith(get_log_file_suffix()):
                print ('Log file must end in %s.',get_log_file_suffix())
                print_help()
        ## plot_chart accpets multiple path_to_logs
        plot_chart(chart_type, path_to_png, path_to_logs)
        # plot_chart_acc(chart_type, path_to_png, path_to_logs)