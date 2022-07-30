import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed( 0 )
sns.set_style( "darkgrid" )

def zoom_factory(ax, max_xlim, max_ylim, base_scale=2.0):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == "up":
            # deal with zoom in
            scale_factor = 1 / base_scale
            x_scale = scale_factor / 2
        elif event.button == "down":
            # deal with zoom out
            scale_factor = base_scale
            x_scale = scale_factor * 2
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        if xdata - new_width * (1 - relx) > max_xlim[0]:
            x_min = xdata - new_width * (1 - relx)
        else:
            x_min = max_xlim[0]
        if xdata + new_width * (relx) < max_xlim[1]:
            x_max = xdata + new_width * (relx)
        else:
            x_max = max_xlim[1]
        if ydata - new_height * (1 - rely) > max_ylim[0]:
            y_min = ydata - new_height * (1 - rely)
        else:
            y_min = max_ylim[0]
        if ydata + new_height * (rely) < max_ylim[1]:
            y_max = ydata + new_height * (rely)
        else:
            y_max = max_ylim[1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.figure.canvas.draw()

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect("scroll_event", zoom_fun)

    # return the function
    return zoom_fun


annotated_csv = "/Users/lselig/Desktop/annotations.csv"
try:
    annotated_df = pd.read_csv(annotated_csv)
except:
    annotated_df = pd.DataFrame(
        {
            "marked_ts": [],
        }
    )

figure_mosaic = """
                AAAAA
                """

fig, axs = plt.subplot_mosaic( figure_mosaic, figsize = (15, 9) )
df = pd.read_parquet("/Users/lselig/Desktop/happy_data/data/pq/ea148700010f/0B09C3CF-C70D-46C4-BE89-E0BD26481621/0B09C3CF-C70D-46C4-BE89-E0BD26481621-HH_010f-20220721_082821/eda.parquet")
axs["A"].plot(df.etime, df.conductance, label = "parquet")
max_xlim = axs["A"].get_xlim()  # get current x_limits to set max zoom out
max_ylim = axs["A"].get_ylim()  # get current y_limits to set max zoom out
f = zoom_factory( axs["A"], max_xlim, max_ylim, base_scale = 1.1 )
clicks = plt.ginput(-1, timeout=0)

clicks = list(np.vstack(clicks)[:, 0])
for i, click in enumerate(clicks):
    event = clicks[i]
    annotated_df = annotated_df.append(
        pd.DataFrame(
            {
                "marked_ts": [event * 24 * 60 * 60],
            }
        )
    )
    annotated_df = annotated_df.reset_index(drop=True)
    annotated_df.to_csv(annotated_csv, index=False)
