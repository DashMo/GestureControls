from tokenize import String
import osascript

class VolumeControl():
    def __init__(self, os : String, loop = 1) -> None:
        """Can control/manage system volume 

        Args:
            os (String): 'mac' or 'win' for choosing operating system
            loop (int, optional): Number of iterations before updating volume again (For opencv applications). Defaults to 1.
        """
        self.os = os # windows or mac
        self.count = 0 # number of iterations since we last changed volume
        self.loop = loop # number of iterations after which we should next change volume
        
    def mac_set_volume(self,vol):
        osascript.osascript(f'set volume output volume {vol}')
        
    def mac_get_volume(self):
        osascript.osascript('output volume of (get volume settings)')
        
        
    def win_set_volume(self,vol):
        pass
    def win_get_volume(self):
        pass
    
    
    def set_volume(self,vol):
        """Wrapper to set volume based on operating system

        Args:
            vol (int): Should be in range 0-100

        Returns:
            int: 0 for success, -1 for failure
        """
        if self.count == 0:
            print("setting volume to",vol)
            if self.os == 'mac':
                self.mac_set_volume(vol)
            elif self.os == 'win':
                self.win_set_volume(vol)
            else:
                print(f'{self.os} is not currently supported for volume control')
                return -1
        self.count = (self.count+1) % self.loop
        return 0
        # Not currently running python 3.10 (not supported on older versions)
        # match self.os:
        #     case 'mac': self.mac_set_volume()
            
        #     case 'win': self.win_set_volume()
            
        #     case _ :    print(f'{self.os} is not currently supported for volume control')
            