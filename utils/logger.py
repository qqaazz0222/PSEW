
from rich.table import Table

def console_banner(console):
    console.clear()
    banner = [
        "\n",
        " ███████▙╗  ▟███████╗  ████████╗  ██╗ ██╗ ██╗",
        " ██╔═══██║  ██╔═════╝  ██╔═════╝  ██║ ██║ ██║",
        " ███████▛║  ▜██████▙╗  ████████╗  ██║ ██║ ██║",
        " ██╔═════╝        ██║  ██╔═════╝  ██║ ██║ ██║",
        " ██║        ███████▛║  ████████╗  ▜███▛▜███▛║",
        " ╚═╝        ╚═══════╝  ╚═══════╝  ╚════╩════╝",
        "  P o s e  S c e n e  E v e r y  W h e r e   "
    ]
    for line in banner:
        console.print(line, style="bold green")

def console_process(console, process_name):
    console.log(f"\n[bold green reverse] ‣ {process_name} [/bold green reverse]")
    
def console_args(console, args):
    console.log("\n[bold green reverse] ∙ Set-Up Arguments [/bold green reverse]")
    table = Table(show_header=True, show_footer=False)
    table.add_column("Argument", width=17)
    table.add_column("Value", width=40)
    for arg, value in args.items():
        table.add_row(arg, str(value))
    console.print(table)
    
def console_videos(console, video_list, cur_video):
    console.log("\n[bold green reverse] ∙ Current Working Video [/bold green reverse]")
    table = Table(show_header=True, show_footer=False)
    table.add_column("Video List", width=60)
    for video in video_list:
        if video == cur_video:
            table.add_row(f"[*] {video}", style="bold green")
        else:
            table.add_row(f"[ ] {video}")
    console.print(table)