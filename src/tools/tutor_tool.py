import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def launch_transformer_tutor(topic: str) -> str:
    """
    Tutor Agent Tool: Launches the interactive Transformer GUI from the professor's class.
    """
    logger.info(f"Tutor Agent invoked Transformer GUI for topic: {topic}")
    
    # Absolute path to the professor's class script
    gui_script = "/Users/gagura/NCKU/AI相關課程/成大AI課程/114-2 ML & DL/chapter8_transformer_gui.py"
    
    if os.path.exists(gui_script):
        try:
            # Launch in the background using subprocess.Popen
            subprocess.Popen(["python3", gui_script], cwd=os.path.dirname(gui_script))
            return f"Tutor Agent: I have launched the interactive Transformer GUI on your screen to dynamically demonstrate '{topic}'. Please check the new window!"
        except Exception as e:
            logger.error(f"Failed to launch GUI: {e}")
            return f"Tutor Agent Error: Failed to launch GUI due to {e}."
    else:
        return f"Tutor Agent Error: Could not find the Transformer GUI script at {gui_script}."
