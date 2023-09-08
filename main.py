from io import StringIO
from src.interface import *
import sys
from src.convert import *

with open("config.json", "r") as f:
    config = json.load(f)


def init_tabs():
    """
    Init the tabs of the interface, to add a new tab, create it in interface.py and add it here
    """

    root = tk.Tk()
    root.title("Image generation")
    root.geometry("800x900")

    tab_main = ttk.Notebook(root)

    tab_generation = ttk.Frame(tab_main)
    tab_controlnet = ttk.Frame(tab_main)
    tab_tools = ttk.Notebook(tab_main)
    tab_train = ttk.Frame(tab_main)

    # main tabs
    tab_main.add(tab_generation, text='Stable diffusion')
    tab_main.add(tab_controlnet, text='Control net')
    tab_main.add(tab_tools, text='Tools')
    tab_main.add(tab_train, text='Train stable diffusion')

    # tools tabs
    tab_yolo = ttk.Frame(tab_tools)
    tab_add = ttk.Frame(tab_tools)
    tab_analyze = ttk.Frame(tab_tools)
    tab_get_random = ttk.Frame(tab_tools)
    tab_run_yolo = ttk.Frame(tab_tools)
    tab_run_statistic = ttk.Frame(tab_tools)
    tab_display_bbox = ttk.Frame(tab_tools)
    tab_data_augmentation = ttk.Frame(tab_tools)
    tab_cosim = ttk.Frame(tab_tools)

    tab_tools.add(tab_yolo, text='Yolo_detected')
    tab_tools.add(tab_add, text='Merge dataset')
    tab_tools.add(tab_analyze, text='Analyse dataset')
    tab_tools.add(tab_get_random, text='Random images')
    tab_tools.add(tab_run_yolo, text='Run Yolo')
    tab_tools.add(tab_run_statistic, text='Display statistic')
    tab_tools.add(tab_display_bbox, text='Display bbox')
    tab_tools.add(tab_data_augmentation, text='Data augmentation')
    tab_tools.add(tab_cosim, text='Cosine similarity')

    # monitor
    monitor = tk.Frame(root)
    monitor.pack_propagate(False)
    monitor.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
    # size of monitor
    monitor.config(width=800, height=300)
    # add scrollbar for monitor
    scrollbar = tk.Scrollbar(monitor)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    # add text widget to monitor
    out = tk.Text(monitor, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    out.configure(background='black', foreground='white', font=("Courier", 10))
    out.pack(expand=True, fill="both")
    # configure the scrollbar
    scrollbar.config(command=out.yview)
    # redirect stderr
    sys.stderr = out

    # redirect stdout
    temp = StringIO()
    sys.stdout = temp
    update_terminal(root, temp, out)

    # tabs methods
    stable_diffusion_generation_tab(tab_generation)
    controlnet_tab(tab_controlnet)
    merge_tab(tab_add)
    analyze_tab(tab_analyze)
    add_random_tab(tab_get_random)
    tools_tab(tab_yolo)
    run_yolo(tab_run_yolo)
    statistic_tab(tab_run_statistic)
    display_bbox_tab(tab_display_bbox)
    data_augmentation_tab(tab_data_augmentation)
    training_tab(tab_train)
    cosim_tab(tab_cosim)

    tab_main.pack(expand=1, fill="both")

    root.mainloop()


if __name__ == "__main__":

    print("Starting")
    init_tabs()
