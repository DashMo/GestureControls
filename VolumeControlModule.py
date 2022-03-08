from tokenize import String
import osascript

class VolumeControl():
    def __init__(self, os : String) -> None:
        self.os = os
        
    def mac_set_volume(vol):
        osascript.osascript(f'set volume output volume {vol}')
    def mac_get_volume():
        osascript.osascript('output volume of (get volume settings)')
        
    def win_set_volume(vol):
        pass
    def win_get_volume():
        pass
    
    def set_volume(vol):
        