# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import imgui
import glfw
import threading
import random
import string
import numpy as np
import time

import sys, os
sys.path += [os.path.abspath(os.path.dirname(__file__) + '/..')]
from dnnlib import EasyDict
from viewer import gl_viewer
from viewer.utils import imgui_item_width, begin_inline

#----------------------------------------------------------------------------
# Helper class for UIs with toolbar on the left and output image on the right

class ToolbarViewer:
    def __init__(self, name, pad_bottom=0, hidden=False, batch_mode=False):
        self.output_key = ''.join(random.choices(string.ascii_letters, k=20))
        self.pad_bottom = pad_bottom
        self.v = gl_viewer.viewer(name, hidden=hidden or batch_mode)
        self.menu_bar_height = 0
        self.toolbar_width = 300
        self.img_shape = [3, 4, 4]
        self.output_pos_tl = np.zeros(2, dtype=np.float32)
        self.output_pos_br = np.zeros(2, dtype=np.float32)
        self.output_area_tl = np.zeros(2, dtype=np.float32)
        self.output_area_br = np.zeros(2, dtype=np.float32)
        self.ui_locked = True
        self.state = EasyDict()
        
        # User-provided
        self.setup_state()

        # User nearest interpolation for sharpness by default
        self.v.set_interp_nearest()
        
        # Batch mode: handle compute loop manually, don't start UI
        if not batch_mode:
            self.start_UI()

    def start_UI(self):
        compute_thread = threading.Thread(target=self._compute_loop, args=[])
        self.v.start(self._ui_main, (compute_thread), self.setup_callbacks)

    @property
    def font_size(self):
        return self.v.font_size

    @property
    def ui_scale(self):
        return self.v.ui_scale

    @property
    def content_rect(self):
        return np.concatenate((self.output_area_tl, self.output_area_br), axis=0)

    @property
    def content_size(self):
        x1, y1, x2, y2 = self.content_rect
        return np.array([x2-x1, y2-y1])

    @property
    def mouse_pos_abs(self):
        return np.array(imgui.get_mouse_pos())

    @property
    def mouse_pos_content_norm(self):
        return (self.mouse_pos_abs - self.output_area_tl) / self.content_size
    
    @property
    def mouse_pos_img_norm(self):
        dims = self.output_pos_br - self.output_pos_tl
        if any(dims == 0):
            return np.array([-1, -1], dtype=np.float32) # no valid content
        return (self.mouse_pos_abs - self.output_pos_tl) / dims

    def _ui_main(self, v):
        self._toolbar_wrapper()
        self._draw_output()

    def _compute_loop(self):
        while not self.v.quit:
            img = self.compute()
            if img is not None:
                H, W, C = img.shape
                self.img_shape = [C, H, W]
                self.v.upload_image(self.output_key, img)
            else:
                time.sleep(1/60)
    
    def _draw_output(self):
        v = self.v

        W, H = glfw.get_window_size(v._window)
        imgui.set_next_window_size(W - self.toolbar_width, H - self.menu_bar_height)
        imgui.set_next_window_position(self.toolbar_width, self.menu_bar_height)

        s = v.ui_scale

        begin_inline('Output')
        BOTTOM_PAD = max(self.pad_bottom, int(round(20 * s)) + 6) # extra user content below image
        rmin, rmax = imgui.get_window_content_region_min(), imgui.get_window_content_region_max()
        cW, cH = [int(r-l) for l,r in zip(rmin, rmax)]
        aspect = self.img_shape[2] / self.img_shape[1]
        out_size = min(cW, aspect*(cH - BOTTOM_PAD))
        
        # Draw provided image
        v.draw_image(self.output_key, width=out_size)
        self.output_pos_tl[:] = imgui.get_item_rect_min()
        self.output_pos_br[:] = imgui.get_item_rect_max()

        # Potential space for content
        self.output_area_tl[:] = self.output_pos_tl
        self.output_area_br[:] = np.array(rmax) - np.array([0, BOTTOM_PAD])

        # Equal spacing
        imgui.columns(2, 'outputBottom', border=False)

        # Extra UI elements below output
        self.draw_output_extra()
        imgui.next_column()

        # Scaling buttons, right-aligned within child
        imgui.begin_child('sizeButtons', width=0, height=0, border=False)
        child_w = imgui.get_content_region_available_width()

        sizes = ['0.5', '1', '2', '3', '4']    
        button_W = 40 * s
        pad_left = max(0, child_w - (button_W * len(sizes)))

        for i, s in enumerate(sizes):
            imgui.same_line(position=pad_left+i*button_W)
            if imgui.button(f'{s}x', width=button_W-4): # tiny pad
                resW = int(self.img_shape[2] * float(s))
                resH = int(self.img_shape[1] * float(s))
                glfw.set_window_size(v._window,
                    width=resW+W-cW, height=resH+H-cH+BOTTOM_PAD)
        imgui.end_child()

        imgui.columns(1)
        self.draw_overlays(imgui.get_window_draw_list())

        imgui.end()

    def _toolbar_wrapper(self):
        if imgui.begin_main_menu_bar():
            self.menu_bar_height = imgui.get_window_height()

            # Right-aligned button for locking / unlocking UI
            T = 'L' if self.ui_locked else 'U'
            C = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
            s = self.v.ui_scale

            imgui.text('') # needed for imgui.same_line

            # UI scale slider
            if not self.ui_locked:
                imgui.same_line(position=imgui.get_window_width()-300-25*s)
                with imgui_item_width(300): # size not dependent on s => prevents slider drift
                    ch, val = imgui.slider_float('', s, 0.5, 2.0)
                if ch:
                    self.v.set_ui_scale(val)

            imgui.same_line(position=imgui.get_window_width()-25*s)
            imgui.push_style_color(imgui.COLOR_TEXT, *C)
            if imgui.button(T, width=20*s):
                self.ui_locked = not self.ui_locked
            imgui.pop_style_color()
            imgui.end_main_menu_bar()

        
        # Constant width, height dynamic based on output window
        v = self.v
        _, H = glfw.get_window_size(v._window)
        self.toolbar_width = 350 * v.ui_scale
        imgui.set_next_window_size(self.toolbar_width, H - self.menu_bar_height)
        imgui.set_next_window_position(0, self.menu_bar_height)

        # User callback
        begin_inline('toolbar')
        self.draw_toolbar()
        imgui.end()

    def mouse_over_image(self):
        x, y = self.mouse_pos_img_norm
        return (0 <= x <= 1) and (0 <= y <= 1)

    def mouse_over_content(self):
        x, y = self.mouse_pos_content_norm
        return (0 <= x <= 1) and (0 <= y <= 1)
    
    #-----------------------------------------------------------------------------------
    # User-provided functions

    # Draw toolbar, must be implemented
    def draw_toolbar(self):
        pass

    # Draw extra UI elements below output image
    def draw_output_extra(self):
        if self.pad_bottom > 0:
            raise RuntimeError('Not implemented')

    # Draw overlays using main window draw list
    def draw_overlays(self, draw_list):
        pass
    
    # Perform computation, returning single np/torch image, or None
    def compute(self):
        pass

    # Program state init
    def setup_state(self):
        pass

    # GLFW callbacks
    def setup_callbacks(self, window):
        pass


#-----------------------------------------------------------------------------
# Example usage

if __name__ == '__main__':
    import torch
    
    class Test(ToolbarViewer):
        def setup_state(self):
            self.state.seed = 0
        
        def compute(self):
            torch.manual_seed(self.state.seed)
            img = torch.randn((256, 256, 3), dtype=torch.float32, device='cuda')
            return img.clip(0, 1) # HWC
        
        def draw_toolbar(self):
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]

    viewer = Test('test_viewer')
    print('Done')
