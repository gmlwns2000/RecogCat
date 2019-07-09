import time
import requests

class CatEvent:
    def __init__(self, string=""):
        spl = string.splitlines()
        self.id = int(spl[0])
        self.error = None
        if spl[1] == "CameraError":
            self.error = "CameraError"
        else:
            self.unlocked = spl[-1] == "Unlocked"
            self.cats = []
            for i in range(1, len(spl)-1):
                line = spl[i]
                if line.startswith("Cat("):
                    line = line[4:-1]
                    items = line.split('|')
                    diff = float(items[0])
                    isSame = items[1]=="True"
                    bbox = items[2]
                    landmarks = items[3]
                    self.cats.append([diff, isSame, bbox, landmarks])
                else:
                    print("unknown object")
            self.hasCat = len(self.cats)>0
    
class Interface:
    def __init__(self):
        self.lastevent = None
        
    def push(self, event):
        lastevent = self.lastevent
        if not lastevent is None:
            if lastevent.unlocked != event.unlocked:
                if event.unlocked:
                    self.onUnlocked(event)
                else:
                    self.onLocked(event)
        else:
            self.lastevent = event
        self.onUpdate(event)
        self.lastevent = event
        
    def onUnlocked(self, event):
        pass
    def onLocked(self, event):
        pass
    def onUpdate(self, event):
        pass

class InterfaceImpl(Interface):
    def __init__(self):
        super().__init__()
    
    def onUnlocked(self, event):
        print("unlocked"*10)
    
    def onLocked(self, event):
        print("locked"*10)
    
    def onUpdate(self, event):
        print(event.unlocked, event.hasCat, event.cats)
    
if __name__ == "__main__":
    site = "http://118.36.38.55:3206/"
    interface = InterfaceImpl()
    last_frame = 0
    while True:
        try:
            contents = requests.get(site, timeout=10)
            event = CatEvent(contents.text)
            interface.push(event)
        except requests.exceptions.ReadTimeout as ex:
            print("read timeout!")
        except requests.exceptions.ConnectTimeout as ex:
            print("server is dead or internet is dead")
        except Exception as ex:
            print("unknown exception", ex)
        time.sleep(max(0.001, 0.1-(time.time()-last_frame)))
        last_frame = time.time()
