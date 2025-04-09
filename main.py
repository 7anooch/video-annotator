#!/usr/bin/env python3
"""
Main entry point for the Video Annotator application.

This script provides a command-line interface for running the different components
of the Video Annotator application.
"""

import argparse
import sys
import os
import tkinter as tk

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Video Annotator")
    parser.add_argument('--mode', type=str, default='annotator',
                        choices=['annotator', 'legacy', 'config', 'plot', 'analyze',
                                'ground_truth', 'export', 'visualize', 'visualize_enhanced',
                                'profile', 'patch_gaps'],
                        help="Mode to run the application in")
    parser.add_argument('--csv', type=str, nargs='*', help="Path to the annotation CSV file(s)")
    parser.add_argument('--video', type=str, help="Path to the video file")
    parser.add_argument('--config', type=str, default='config.json',
                        help="Path to the configuration file")
    parser.add_argument('--side_controls', action='store_true', default=False,
                        help="Place controls on the right side")
    parser.add_argument('--advanced', action='store_true', default=False,
                        help="Run advanced analyses (only applicable in analyze mode)")

    args = parser.parse_args()

    # Run the selected mode
    if args.mode == 'annotator':
        from annotator_modular import main as annotator_main
        result = annotator_main()
        if result:
            root, _, _, _ = result
            root.mainloop()
    elif args.mode == 'legacy':
        from annotator import main as legacy_main
        legacy_main()
        tk.mainloop()
    elif args.mode == 'config':
        from src.ui.config_editor import main as config_main
        config_main()
    elif args.mode == 'plot':
        try:
            from src.tools.analysis.visualization import main as plot_main
        except ImportError:
            from src.tools.plot import main as plot_main
        plot_main()
    elif args.mode == 'analyze':
        # Import from the new location, but fall back to the old location if not found
        try:
            from src.tools.analysis.analyze import main as analyze_main
        except ImportError:
            from src.tools.analyze import main as analyze_main
        analyze_main(csv_paths=args.csv, advanced=args.advanced)
    elif args.mode == 'ground_truth':
        from src.tools.gen_ground_truth import main as ground_truth_main
        ground_truth_main()
    elif args.mode == 'export':
        from src.tools.export_gui import main as export_main
        export_main()
    elif args.mode == 'visualize':
        try:
            from src.tools.analysis.visualization import main as visualize_main
        except ImportError:
            from src.tools.visualization import main as visualize_main
        visualize_main()
    elif args.mode == 'visualize_enhanced':
        try:
            from src.tools.analysis.visualization_enhanced import main as visualize_enhanced_main
        except ImportError:
            from src.tools.visualization_enhanced import main as visualize_enhanced_main
        visualize_enhanced_main()
    elif args.mode == 'profile':
        from src.tools.performance_profiler import main as profile_main
        profile_main()
    elif args.mode == 'patch_gaps':
        from src.tools.patch_gaps import main as patch_gaps_main
        patch_gaps_main()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
