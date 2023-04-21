import logging
import os
import scipy
import numpy as np
import pandas as pd
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider, RangeSlider, Spinner, Switch
from bokeh.plotting import figure, save
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models.widgets import Button

from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import ColumnDataSource, Div, HoverTool, Select, Tabs, Toggle, CheckboxGroup, RadioButtonGroup
from bokeh.models import RangeSlider, Label, PreText, BoxSelectTool, LassoSelectTool
from bokeh.models.mappers import LinearColorMapper
from bokeh.plotting import figure

from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.utils import find_files_by_extension, make_directory


WIDTH = 300
HEIGHT = 300

# Find all files
folder_path = "./raw/ReachingData"
file_paths = list(find_files_by_extension(folder_path, extension=".mat"))

# Make a bokeh select widget
select = Select(title="File:", value=file_paths[0], options=file_paths)

def make_layout():
    # Make function to load the data
    def load_data(file_path):
        logging.info(f"Processing file: {file_path}")
        mat_dict = scipy.io.loadmat(file_path)

        # behavior
        t, x, y, vx, vy, ax, ay = mat_dict['data'][0][0][2][0, 0]
        df_behavior = pd.DataFrame(
            {"time": t[:, 0], "x": x[:, 0] + 3, "y": y[:, 0] + 33, "vx": vx[:, 0], "vy": vy[:, 0], "ax": ax[:, 0], "ay": ay[:, 0]}
            )

        # trial table
        keys = mat_dict['data'][0][0][1][0].dtype.names
        values = mat_dict['data'][0][0][1][0][0]

        if 'CO' in file_path:
            experiment_type = 'CO'
            start_time = values['startTime'][:, 0]
            go_cue_time = values['goCueTime'][:, 0]
            target_on_time = values['tgtOnTime'][:, 0]
            end_time = values['endTime'][:, 0]
            target_id = values['tgtID'][:, 0]
            result = values['result'][:]
            target_corners = values['tgtCorners'] # left, top, right, bottom
            return_end_time = np.append(values['tgtOnTime'][1:, 0], values['tgtOnTime'][-1, 0]+1.)
            hold_period = go_cue_time - target_on_time
            reach_period = end_time - go_cue_time
            return_period = return_end_time - end_time
            df_trials = pd.DataFrame({"start_time": start_time, "go_cue_time": go_cue_time, "target_on_time": target_on_time,
                                      "end_time": end_time, "return_end_time": return_end_time, "target_id": target_id, "result": result, "hold_period": hold_period, 
                                      "reach_period": reach_period, "return_period": return_period})

            # add colors to trajectory
            t = t[:, 0]
            # use matplotlib's tab10
            colors = ['#f5852a', '#1a2feb', '#09ab2c',]
            c = np.array(['black' for _ in range(len(t))])
            for (start, go_cue, target_on, end) in zip(start_time, go_cue_time, target_on_time, end_time):
                if not np.isnan(start):
                    c[(t >= start) & (t <= target_on)] = colors[0]
                    c[(t >= target_on) & (t <= go_cue)] = colors[1]
                    c[(t >= go_cue) & (t <= end)] = colors[2]
            df_behavior['c'] = c

            # make targets
            target = {'left': [], 'top': [], 'right': [], 'bottom': [], 'start_time':[], 'end_time':[]}
            for i, (start, end) in enumerate(zip(target_on_time, end_time)):
                left, top, right, bottom = target_corners[i]
                # check if nan
                if not np.isnan(left):
                    target['left'].append(left)
                    target['top'].append(top)
                    target['right'].append(right)
                    target['bottom'].append(bottom)
                    target['start_time'].append(start)
                    target['end_time'].append(end)
            df_target = pd.DataFrame(target)

        else:
            experiment_type = 'RT'
            start_time = values['startTime'][:, 0]
            go_cue_time = values['goCueTime'][:]
            end_time = values['endTime'][:, 0]
            target_id = values['tgtID'][:, 0]
            result = values['result'][:]
            target_center = values['tgtCtr'] # (8)
            target_size = values['tgtSize']
            num_attempted = values['numAttempted'][:, 0]
            df_trials = pd.DataFrame({"start_time": start_time, "end_time": end_time, "target_id": target_id, 
                                      "result": result,  "num_targets": num_attempted})

            # add colors to trajectory
            t = t[:, 0]
            # use matplotlib's tab10
            colors = ['#f5852a', '#1a2feb', '#09ab2c', "#34deeb", "#b434eb"]
            c = np.array(['black' for _ in range(len(t))])
            for (start, go_cue, end) in zip(start_time, go_cue_time, end_time):
                go_cue_1 = go_cue[0]
                if not np.isnan(go_cue_1):
                    c[(t >= start) & (t <= go_cue_1)] = colors[0]
                    c[((t >= go_cue_1) & (t <= end))] = colors[4]
                    go_cue_2 = go_cue[1]
                    if not np.isnan(go_cue_2):
                        c[(t >= go_cue_1) & (t <= go_cue_2)] = colors[1]
                        go_cue_3 = go_cue[2]
                        if not np.isnan(go_cue_3):
                            c[(t >= go_cue_2) & (t <= go_cue_3)] = colors[2]
                            go_cue_4 = go_cue[3]
                            if not np.isnan(go_cue_4):
                                c[(t >= go_cue_3) & (t <= go_cue_4)] = colors[3]
            df_behavior['c'] = c

            # make targets
            target = {'left': [], 'top': [], 'right': [], 'bottom': [], 'start_time':[], 'end_time':[]}
            for i, (start, end) in enumerate(zip(start_time, end_time)):
                x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = target_center[i]
                xs = ys = target_size[i]
                # check if nan
                for x, y in [(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]:
                    if not np.isnan(x):
                        target['left'].append(x - xs/2)
                        target['top'].append(y + ys/2)
                        target['right'].append(x + xs/2)
                        target['bottom'].append(y - ys/2)
                        target['start_time'].append(start)
                        target['end_time'].append(end)
            df_target = pd.DataFrame(target)

        # neural data
        # skip for now
        return experiment_type, df_behavior, df_target, df_trials

    experiment_type, df_behavior, df_target, df_trials = load_data(select.value)

    def get_range(series, pad=True):
        # instead of using min and max, use the 5th and 95th percentile
        x_range = (series.quantile(0.01), series.quantile(0.99))
        if pad:
            x_range_pad = (x_range[1] - x_range[0]) * 0.05
            x_range = (x_range[0] - x_range_pad, x_range[1] + x_range_pad)
        return x_range


    def make_2d_trajectory_plot(df, spinners, x='x', y='y', c='c', time='time', title=None):
        source = ColumnDataSource({"x": df[x][df[time] <= 4], "y": df[y][df[time] <= 4], "c": df[c][df[time] <= 4]})

        tools = "pan,wheel_zoom,box_zoom,reset"
        plot = figure(width=WIDTH, height=HEIGHT, toolbar_sticky=False, tools=tools, title=title,
                        x_range=get_range(df[x]), y_range=get_range(df[y]))
        plot.title.text_font_size = "15px"
        # plot_long_term.scatter("x", "y", color="black", source=source_behavior, alpha=0.4, legend_group='label')
        plot.circle("x", "y", color="c", source=source)
        # plot.line("x", "y", color="black", source=source, alpha=0.5)
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.toolbar.logo = None
        # plot_pos.legend.padding = 4
        # plot_pos.legend.margin = 5
        # plot_pos.legend.label_standoff = 2
        # plot_pos.legend.spacing = 2
        # plot_pos.legend.border_line_color  = "black"
        # plot_pos.legend.background_fill_alpha = 0.8
        # plot_pos.legend.orientation = "horizontal"
        # plot_pos.toolbar_location = None
        # plot_pos.title.text_font_style = "normal"

        # Convert the DataFrame to a dictionary and then to a JSON string
        df_json = df.to_json(orient="records")

        # Define the JavaScript callback
        callback = CustomJS(
            args=dict(source=source, df_json=df_json, x=x, y=y, c=c, time=time, spinner_start=spinners[0], spinner_end=spinners[1],),
            code="""
            const start = spinner_start.value;
            const end = spinner_end.value;
            const df = JSON.parse(df_json);
            const new_data = df.filter(row => row[time] >= start && row[time] <= end);

            let x_list = [];
            let y_list = [];
            let c_list = [];
            for (const row of new_data) {
                x_list.push(row[x]);
                y_list.push(row[y]);
                c_list.push(row[c]);
            }

            if (x_list.length > 0) {
                c_list.pop();
                c_list.push('red');
            }
            
            source.data = {x: x_list, y: y_list, c: c_list};
            source.change.emit();
        """,
        )

        # Add the callback to the slider
        spinners[0].js_on_change("value", callback)
        spinners[1].js_on_change("value", callback)
        return plot


    def add_squares_to_plot(plot, df, spinners):
        mask = df['end_time'] < 4
        source = ColumnDataSource({"left": df['left'][mask], "top": df['top'][mask], 
                                "right": df['right'][mask], "bottom": df['bottom'][mask]})
        patch = plot.quad(top='top', bottom='bottom', left='left', right='right', color="black", alpha=0.5, source=source)
        patch.level = 'underlay'

        # Convert the DataFrame to a dictionary and then to a JSON string
        df_json = df.to_json(orient="records")

        # Define the JavaScript callback
        callback = CustomJS(
            args=dict(source=source, df_json=df_json, time='time', spinner_start=spinners[0], spinner_end=spinners[1]),
            code="""
            const start = spinner_start.value;
            const end = spinner_end.value;
            const df = JSON.parse(df_json);
            const new_data = df.filter(row => row.end_time >= start && row.start_time <= end);

            let left = [];
            let top = [];
            let right = [];
            let bottom = [];
            for (const row of new_data) {
                left.push(row.left);
                top.push(row.top);
                right.push(row.right);
                bottom.push(row.bottom);
            }
            
            source.data = {left: left, top: top, right: right, bottom: bottom};
            source.change.emit();
        """,
        )

        # Add the callback to the slider
        spinners[0].js_on_change("value", callback)
        spinners[1].js_on_change("value", callback)


    # Create a range slider 
    spinner_start = Spinner(title="Start (s)", low=df_behavior.time.min(), high=df_behavior.time.max(), step=1.0, value=df_behavior.time.min(), width=80)
    spinner_end = Spinner(title="End (s)", low=df_behavior.time.min(), high=df_behavior.time.max(), step=1.0, value=df_behavior.time.max(), width=80)
    
    lock_start_switch = Switch(active=False)
    # slider = RangeSlider(start=df_behavior.time.min(), end=df_behavior.time.max(), 
    #                     value=(df_behavior.time.min(), df_behavior.time.min() + 4),
    #                     step=0.1, title="Time (s)", width=1000)

    plot_pos = make_2d_trajectory_plot(df_behavior, x='x', y='y', time='time', title='Hand Position', spinners=(spinner_start, spinner_end))
    plot_vel = make_2d_trajectory_plot(df_behavior, x='vx', y='vy', time='time', title='Hand Velocity', spinners=(spinner_start, spinner_end))
    add_squares_to_plot(plot_pos, df_target, spinners=(spinner_start, spinner_end))


    # Create spinner to jump between trials
    spinner = Spinner(title="Trial #", low=0, high=len(df_trials), step=1, value=0, width=80)
    trial_description = Div(text='', width=300)

    df_trials_json = df_trials.to_json(orient="records")
    spinner_callback = CustomJS(
        args=dict(spinner_start=spinner_start, spinner_end=spinner_end, df_json=df_trials_json, trial_description=trial_description, experiment_type=experiment_type),
        code="""
        const trial = parseInt(cb_obj.value);
        const df = JSON.parse(df_json);
        const row = df[trial];

        if (experiment_type == 'CO') {
            spinner_start.value = row.target_on_time;
            spinner_end.value = row.return_end_time;
            trial_description.text = `<p><b>Target ID</b>: ${row.target}</p><p><b>Result</b>: ${row.result}</p><p><b>Hold period</b>: ${row.hold_period}</p><p><b>Reach period</b>: ${row.reach_period}</p><p><b>Return period</b>: ${row.return_period}</p>`;
        } else {
            console.log(row);
            spinner_start.value = row.start_time;
            spinner_end.value = row.end_time;
            trial_description.text = `<p><b>Target ID</b>: ${row.target}</p><p><b>Result</b>: ${row.result}</p><p><b>Trial length</b>: ${row.end_time - row.start_time}</p><p><b>Number of targets</b>: ${row.num_targets}</p>`;
        }
    """,
    )
    spinner.js_on_change("value", spinner_callback)


    # Create a play button
    play_button = Button(label="►", width=40, height=40)
    # Create a row with the play button and the slider
    controls = row(play_button, column(spinner_start, lock_start_switch), spinner_end)


    play_callback = CustomJS(
        args=dict(play_button=play_button, spinner_start=spinner_start, spinner_end=spinner_end, lock_start_switch=lock_start_switch),
        code="""
        function update() {
            if (!lock_start_switch.active) {
                const start = spinner_start.value;
                spinner_start.value = start + 0.1;
            }
            const end = spinner_end.value;
            spinner_end.value = end + 0.1;
        }

        if (play_button.label == "►") {
            play_button.label = "⏸";

            play_button.timer = setInterval(update, 10);
        } else {
            play_button.label = "►";
            if (play_button.timer) {
                clearInterval(play_button.timer);
                play_button.timer = null;
            }
        }
    """,
    )

    # Add the callback to the play button
    play_button.js_on_click(play_callback)

    # Display the scatter plot and slider
    # plot_mice = gridplot([[plot_mice]], sizing_mode='stretch_both')
    layout = column(select, controls, row(plot_pos, plot_vel), spinner, trial_description)
    return layout
    
og_layout = make_layout()

def update_layout(attr, old, new):
    og_layout.children = make_layout().children
select.on_change('value', update_layout)

curdoc().add_root(og_layout)
curdoc().title = "NHPData"
