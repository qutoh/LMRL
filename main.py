#!/usr/bin/env python3
import traceback
import tcod
import argparse

# Assuming the script is run from the project root
from core.common.config_loader import config
from core.ui.ui_manager import UIManager
from core.llm.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(description="Run the AI Storyteller Engine.")
    parser.add_argument('--load', type=str, help='Path to a saved run directory to load.')
    args = parser.parse_args()

    font_path = "res/dejavu10x10_gs_tc.png"
    tileset = tcod.tileset.load_tilesheet(font_path, 32, 8, tcod.tileset.CHARMAP_TCOD)

    # --- Adaptive Resolution Logic ---
    if config.settings.get("ADAPTIVE_RESOLUTION_MODE", False):
        screen_width_pixels, screen_height_pixels = tcod.sys_get_current_resolution()
        console_width = screen_width_pixels // tileset.tile_width
        console_height = screen_height_pixels // tileset.tile_height
        sdl_window_flags = tcod.context.SDL_WINDOW_FULLSCREEN_DESKTOP

        # Update the in-memory config before anything else uses it
        config.settings["MAP_WIDTH"] = console_width
        config.settings["MAP_HEIGHT"] = console_height
        print(f"[SYSTEM] Adaptive resolution enabled. Map size set to {console_width}x{console_height}.")

    else:
        console_width = config.settings.get("MAP_WIDTH", 120)
        console_height = config.settings.get("MAP_HEIGHT", 80)
        sdl_window_flags = tcod.context.SDL_WINDOW_RESIZABLE
    # --- End Adaptive Resolution Logic ---

    model_manager = ModelManager()

    try:
        with tcod.context.new(
                columns=console_width,
                rows=console_height,
                tileset=tileset,
                title="AI Storyteller",
                vsync=True,
                sdl_window_flags=sdl_window_flags
        ) as context:
            root_console = tcod.console.Console(console_width, console_height, order="F")
            ui_manager = UIManager(context, root_console, args.load, model_manager, tileset)
            ui_manager.run()

    except Exception:
        # Print the error to the console and write it to a log file.
        tb_str = traceback.format_exc()
        with open("crash.log", "w") as f:
            f.write(tb_str)
        print("\n---FATAL CRASH---")
        print(tb_str)
        print("Crash details have been saved to crash.log")
        # Keep the window open so the user can see the traceback
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()