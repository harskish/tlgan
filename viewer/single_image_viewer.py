# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

from pathlib import Path
from threading import Thread
import multiprocessing as mp
import numpy as np
import random
import string
import time
import glfw
import imgui
import ctypes
import torch

import sys
sys.path += [str(Path(__file__).parents[1])]
from viewer.gl_viewer import viewer
from viewer.utils import begin_inline

class ImgShape(ctypes.Structure):
    _fields_ = [('h', ctypes.c_uint), ('w', ctypes.c_uint), ('c', ctypes.c_uint)]
class WindowSize(ctypes.Structure):
    _fields_ = [('w', ctypes.c_uint), ('h', ctypes.c_uint)]

# Single global instance
# Removes need to pass variable around in code
# Just call draw() (optionally call init first)
inst = None

def init(*args, **kwargs):
    global inst
    if inst is None:
        inst = SingleImageViewer(*args, **kwargs)

def draw(*args, **kwargs):
    init('SIV') # no-op if init already performed
    inst.draw(*args, **kwargs)

class SingleImageViewer:
    def __init__(self, title, key=None, dtype='uint8', vsync=True, hidden=False):
        self.title = title
        self.key = key or ''.join(random.choices(string.ascii_letters, k=100))
        self.ui_process = None
        self.vsync = vsync

        # Images are copied to minimize critical section time.
        # With uint8 (~4x faster copies than float32), this is
        # faster than waiting for OpenGL upload (for some reason...)
        self.dtype = dtype

        # Shared resources for inter-process communication
        # One shared 8k rgb buffer allocated (max size), subset written to
        # Size does not affect performance, only memory usage
        self.max_size = (2*3840, 2*2160, 3)
        ctype = ctypes.c_uint8 if self.dtype == 'uint8' else ctypes.c_float
        self.shared_buffer = mp.Array(ctype, np.prod(self.max_size).item())
        
        # Non-scalar type: not updated in single transaction
        # Protected by shared_buffer's lock
        self.latest_shape = mp.Value(ImgShape, *(0,0,0), lock=False)
        
        # Current window size, protected by lock
        self.curr_window_size = mp.Value(WindowSize, *(0,0))
        
        # Scalar values updated atomically
        self.should_quit = mp.Value('i', 0)
        self.has_new_img = mp.Value('i', 0)
        
        # For hiding/showing window
        self.hidden = mp.Value(ctypes.c_bool, hidden, lock=False)

        # Pausing (via pause key on keyboard) speeds up computation
        self.paused = mp.Value(ctypes.c_bool, False, lock=False)
        
        # For waiting until process has started
        self.started = mp.Value(ctypes.c_bool, False, lock=False)

        self._start()

    # Called from main thread, waits until viewer is visible
    def wait_for_startup(self, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.started.value:
            time.sleep(1/10)

    # Called from main thread, loops until viewer window is closed
    def wait_for_close(self):
        while self.ui_process.is_alive():
            time.sleep(0.2)
    
    @property
    def curr_shape(self):
        with self.shared_buffer.get_lock(): # shared lock
            return (self.latest_shape.h, self.latest_shape.w, self.latest_shape.c)

    @property
    def window_size(self):
        with self.curr_window_size.get_lock():
            return (self.curr_window_size.w, self.curr_window_size.h)

    def hide(self):
        self.hidden.value = True

    def show(self, sync=False):
        self.hidden.value = False
        if sync:
            self.wait_for_startup()

    def _start(self):
        self.started.value = False
        self.ui_process = mp.Process(target=self.process_func)
        self.ui_process.start()

    def restart(self):
        self._start()

    def close(self):
        self.should_quit.value = 1
        self.ui_process.join()

    def process_func(self):
        v = viewer(self.title, swap_interval=int(self.vsync), hidden=self.hidden.value)
        v._window_hidden = self.hidden.value
        v.set_interp_nearest()
        compute_thread = Thread(target=self.compute, args=[v])
        v.start(self.ui, [compute_thread], self.set_glfw_callbacks)

    def set_glfw_callbacks(self, window):
        self.window_size_callback(window, *glfw.get_window_size(window)) # set defaults
        glfw.set_window_size_callback(window, self.window_size_callback)

    def window_size_callback(self, window, w, h):
        with self.curr_window_size.get_lock():
            self.curr_window_size.w = w
            self.curr_window_size.h = h

    # Called from main thread
    def draw(self, img_hwc, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        if torch.is_tensor(img_hwc):
            img_hwc = img_hwc.detach().cpu().numpy()

        sz = np.prod(img_hwc.shape)
        assert sz <= np.prod(self.max_size), f'Image too large, max size {self.max_size}'
        
        target_type = np.uint8 if self.dtype == 'uint8' else np.float32
        assert img_hwc.dtype == target_type, f'Expected {self.dtype}, got {img_hwc.dtype}'

        if img_hwc.ndim == 2:
            img_hwc = np.expand_dims(img_hwc, -1)

        # Synchronize
        with self.shared_buffer.get_lock():
            arr_np = np.frombuffer(self.shared_buffer.get_obj(), dtype=self.dtype)
            arr_np[:sz] = np.clip(img_hwc, 0, 255).reshape(-1)
            self.latest_shape.h = img_hwc.shape[0]
            self.latest_shape.w = img_hwc.shape[1]
            self.latest_shape.c = img_hwc.shape[2]
            self.has_new_img.value = 1

    # Called in loop from ui thread
    def ui(self, v):
        if self.should_quit.value == 1:
            glfw.set_window_should_close(v._window, True)
            return
        
        # Visibility changed
        if self.hidden.value != v._window_hidden:
            v._window_hidden = self.hidden.value
            if v._window_hidden:
                glfw.hide_window(v._window)
            else:
                glfw.show_window(v._window)

        if v.keyhit(glfw.KEY_PAUSE):
            self.paused.value = not self.paused.value

        imgui.set_next_window_size(*glfw.get_window_size(v._window))
        imgui.set_next_window_position(0, 0)
        begin_inline('Output')
        
        v.draw_image(self.key, width='fit')
        if self.paused.value:
            imgui.push_font(v._imgui_fonts[30])
            dl = imgui.get_window_draw_list()
            dl.add_rect_filled(5, 8, 115, 43, imgui.get_color_u32_rgba(0,0,0,1))
            dl.add_text(20, 10, imgui.get_color_u32_rgba(1,1,1,1), 'PAUSED')
            imgui.pop_font()
            time.sleep(1/20) # <= real speedup of pausing comes from here

        imgui.end()

    # Called in loop from compute thread
    def compute(self, v):
        self.started.value = True
        while not v.quit:
            if self.has_new_img.value == 1:
                with self.shared_buffer.get_lock():
                    shape = self.curr_shape
                    img = np.frombuffer(self.shared_buffer.get_obj(), dtype=self.dtype, count=np.prod(shape)).copy()
                    self.has_new_img.value = 0

                img = img.reshape(shape)
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                
                v.upload_image_np(self.key, img)
            elif self.paused.value:
                time.sleep(1/10) # paused
            else:
                time.sleep(1/80) # idle