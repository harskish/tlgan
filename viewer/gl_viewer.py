# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# Original by Pauli Kemppinen (https://github.com/msqrt)

"""Imgui viewer that supports separate ui and compute threads, image uploads from torch tensors."""

import numpy as np
import multiprocessing as mp
from pathlib import Path
from urllib.request import urlretrieve
import os

import imgui.core
from imgui.integrations.glfw import GlfwRenderer

import glfw
glfw.ERROR_REPORTING = 'raise' # make sure errors don't get swallowed

import OpenGL.GL as gl
import torch

has_pycuda = False
try:
    import pycuda
    import pycuda.gl as cuda_gl
    import pycuda.tools
    has_pycuda = True
except Exception:
    print('PyCUDA with GL support not available, images will be uploaded from RAM.')

class _texture:
    '''
    This class maps torch tensors to gl textures without a CPU roundtrip.
    '''
    def __init__(self, min_mag_filter=gl.GL_LINEAR):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex) # need to bind to modify
        # sets repeat and filtering parameters; change the second value of any tuple to change the value
        for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, min_mag_filter), (gl.GL_TEXTURE_MAG_FILTER, min_mag_filter)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, *params)
        self.mapper = None
        self.shape = [0,0]

    # be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
    def __del__(self):
        gl.glDeleteTextures(1, [self.tex])
        if self.mapper is not None:
            self.mapper.unregister()

    def set_interp(self, key, val):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, key, val)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_np(self, image):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # support for shapes (h,w), (h,w,1), (h,w,3) and (h,w,4)
        if len(image.shape) == 2:
            image = image.unsqueeze(-1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1) #image.repeat(1,1,3)
        if image.shape[2] == 3:
            image = np.concatenate([image, np.ones_like(image[:,:,0:1])], axis=-1)

        shape = image.shape
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        if shape[0] != self.shape[0] or shape[1] != self.shape[1]:
            # Reallocate
            self.shape = shape
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, shape[1], shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, image)
        else:
            # Overwrite
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, image)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_torch(self, img):
        assert img.device.type == "cuda", "Please provide a CUDA tensor"
        assert img.ndim == 3, "Please provide a HWC tensor"
        assert img.shape[2] < min(img.shape[0], img.shape[1]), "Please provide a HWC tensor"

        if not img.dtype.is_floating_point:
            img = img.to(torch.float32) / 255.0

        # support for shapes (h,w), (h,w,1), (h,w,3) and (h,w,4)
        if img.shape[2] == 1:
            img = img.repeat(1,1,3)
        if img.shape[2] == 3:
            img = torch.cat((img, torch.ones_like(img[:,:,0:1])), 2)
        
        img = img.contiguous()
        if has_pycuda:
            self.upload_ptr(img.data_ptr(), img.shape)
        else:
            self.upload_np(img.detach().cpu().numpy())

    # Copy from cuda pointer
    def upload_ptr(self, ptr, shape):
        assert has_pycuda, 'PyCUDA-GL not available, cannot upload using raw pointer'
        assert shape[-1] == 4, 'Data format not RGBA'

        # reallocate if shape changed or data type changed from np to torch
        if shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.mapper is None:
            self.shape = shape
            if self.mapper is not None:
                self.mapper.unregister()
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, shape[1], shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self.mapper = cuda_gl.RegisteredImage(int(self.tex), gl.GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.WRITE_DISCARD)
        
        # map texture to cuda ptr
        tex_data = self.mapper.map()
        tex_arr = tex_data.array(0, 0)

        # Cast to python integer type
        ptr_int = int(ptr)
        assert ptr_int == ptr, 'Device pointer overflow'

        # copy from torch tensor to mapped gl texture (avoid cpu roundtrip)
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(ptr_int)
        cpy.set_dst_array(tex_arr)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = 4*shape[1]*shape[2]
        cpy.height = shape[0]
        cpy(aligned=False)

        # cleanup
        tex_data.unmap()
        torch.cuda.synchronize()

class _editable:
    def __init__(self, name, ui_code = '', run_code = ''):
        self.name = name
        self.ui_code = ui_code if len(ui_code)>0 else 'imgui.begin(\'Test\')\nimgui.text(\'Example\')#your code here!\nimgui.end()'
        self.tentative_ui_code = self.ui_code
        self.run_code = run_code
        self.run_exception = ''
        self.ui_exception = ''
        self.ui_code_visible = False
    def try_execute(self, string, **kwargs):
        try:
            for key, value in kwargs.items():
                locals()[key] = value
            exec(string)
        except Exception as e: # while generally a bad idea, here we truly want to skip any potential error to not disrupt the worker threads
            return 'Exception: ' + str(e)
        return ''
    def loop(self, v):
        imgui.begin(self.name)
        
        self.run_code = imgui.input_text_multiline('run code', self.run_code, 2048)[1]
        if len(self.run_exception)>0:
            imgui.text(self.run_exception)

        _, self.ui_code_visible = imgui.checkbox('Show UI code', self.ui_code_visible)
        if self.ui_code_visible:
            self.tentative_ui_code = imgui.input_text_multiline('ui code', self.tentative_ui_code, 2048)[1]
            if imgui.button('Apply UI code'):
                self.ui_code = self.tentative_ui_code
            if len(self.ui_exception)>0:
                imgui.text(self.ui_exception)
                
        imgui.end()

        self.ui_exception = self.try_execute(self.ui_code, v=v)

    def run(self, **kwargs):
        self.run_exception = self.try_execute(self.run_code, **kwargs)


class viewer:
    def __init__(self, title, inifile=None, swap_interval=0, hidden=False):
        self.quit = False

        self._images = {}
        self._editables = {}
        self.tex_interp_mode = gl.GL_LINEAR
        
        fname = inifile or "".join(c for c in title.lower() if c.isalnum())
        self._inifile = Path(fname).with_suffix('.ini')

        glfw.init()
        try:
            with open(self._inifile, 'r') as file:
                self._width, self._height = [int(i) for i in file.readline().split()]
                self.window_pos = [int(i) for i in file.readline().split()]
                start_maximized = int(file.readline().rstrip())
                self.ui_scale = float(file.readline().rstrip())
                self.fullscreen = bool(int(file.readline().rstrip()))
                key = file.readline().rstrip()
                while key is not None and len(key)>0:
                    code = [None, None]
                    for i in range(2):
                        lines = int(file.readline().rstrip())
                        code[i] = '\n'.join((file.readline().rstrip() for _ in range(lines)))
                    self._editables[key] = _editable(key, code[0], code[1])
                    key = file.readline().rstrip()
        except Exception as e:
            self._width, self._height = 1280, 720
            self.window_pos = (50, 50)
            self.ui_scale = 1.0
            self.fullscreen = False
            start_maximized = 0

        glfw.window_hint(glfw.MAXIMIZED, start_maximized)
        glfw.window_hint(glfw.VISIBLE, not hidden)

        if self.fullscreen:
            monitor = glfw.get_monitors()[0]
            params = glfw.get_video_mode(monitor)
            self._window = glfw.create_window(params.size.width, params.size.height, title, monitor, None)
        else:
            self._window = glfw.create_window(self._width, self._height, title, None, None)
        glfw.set_window_pos(self._window, *self.window_pos)
        glfw.make_context_current(self._window)

        self._cuda_context = None
        if has_pycuda:
            pycuda.driver.init()
            self._cuda_context = pycuda.gl.make_context(pycuda.driver.Device(0))
        glfw.swap_interval(swap_interval) # should increase on high refresh rate monitors
        glfw.make_context_current(None)

        # TERO
        self._imgui_context = imgui.create_context()
        font = self.get_default_font()
        font_sizes = {int(size) for size in range(10, 31)}
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, size) for size in font_sizes}

        self._context_lock = mp.Lock()

    def get_default_font(self):
        return str(Path(__file__).parent / 'roboto_mono.ttf')
    
    def push_context(self):
        if has_pycuda:
            self._cuda_context.push()
    
    def pop_context(self):
        if has_pycuda:
            self._cuda_context.pop()

    def _lock(self):
        self._context_lock.acquire()
        try:
            glfw.make_context_current(self._window)
        except Exception as e:
            print(str(e))
            self._context_lock.release()
            return False
        return True

    def _unlock(self):
        glfw.make_context_current(None)
        self._context_lock.release()

    # Scales fonts and sliders/etc
    def set_ui_scale(self, scale):
        self.set_font_size(15*scale)
        self.ui_scale = self.font_size / 15

    def set_interp_linear(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.tex_interp_mode = gl.GL_LINEAR

    def set_interp_nearest(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        self.tex_interp_mode = gl.GL_NEAREST

    def editable(self, name, **kwargs):
        if name not in self._editables:
            self._editables[name] = _editable(name)
        self._editables[name].run(**kwargs)

    def keydown(self, key):
        return key in self._pressed_keys

    def keyhit(self, key):
        if key in self._hit_keys:
            self._hit_keys.remove(key)
            return True
        return False

    def draw_image(self, name, scale=1, width=None, pad_h=0, pad_v=0):
        if name in self._images:
            img = self._images[name]
            if width == 'fill':
                scale = imgui.get_window_content_region_width() / img.shape[1]
            elif width == 'fit':
                H, W = img.shape[0:2]
                cW, cH = [r-l for l,r in zip(
                    imgui.get_window_content_region_min(), imgui.get_window_content_region_max())]
                scale = min((cW-pad_h)/W, (cH-pad_v)/H)
            elif width is not None:
                scale = width / img.shape[1]
            imgui.image(img.tex, img.shape[1]*scale, img.shape[0]*scale)

    def close(self):
        glfw.set_window_should_close(self._window, True)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.3) # 0.4

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.set_fullscreen(self.fullscreen)

    def set_fullscreen(self, value):
        monitor = glfw.get_monitors()[0]
        params = glfw.get_video_mode(monitor)
        if value:
            # Save size and pos
            self._width, self._height = glfw.get_window_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)
            glfw.set_window_monitor(self._window, monitor, \
                0, 0, params.size.width, params.size.height, params.refresh_rate)
        else:
            # Restore previous size and pos
            posy = max(10, self.window_pos[1]) # title bar at least partially visible
            glfw.set_window_monitor(self._window, None, \
                self.window_pos[0], posy, self._width, self._height, params.refresh_rate)

    def set_default_style(self, color_scheme='dark', spacing=9, indent=23, scrollbar=27):
        s = imgui.get_style()
        s.window_padding        = [spacing, spacing]
        s.item_spacing          = [spacing, spacing]
        s.item_inner_spacing    = [spacing, spacing]
        s.columns_min_spacing   = spacing
        s.indent_spacing        = indent
        s.scrollbar_size        = scrollbar
        s.frame_padding         = [4, 3]
        s.window_border_size    = 1
        s.child_border_size     = 0
        s.popup_border_size     = 0
        s.frame_border_size     = 0
        s.window_rounding       = 0
        s.child_rounding        = 0
        s.popup_rounding        = 0
        s.frame_rounding        = 0
        s.scrollbar_rounding    = 0
        s.grab_rounding         = 0

        getattr(imgui, f'style_colors_{color_scheme}')(s)
        c0 = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        c1 = s.colors[imgui.COLOR_FRAME_BACKGROUND]
        s.colors[imgui.COLOR_POPUP_BACKGROUND] = [x * 0.7 + y * 0.3 for x, y in zip(c0, c1)][:3] + [1]

    def start(self, loopfunc, workers = (), glfw_init_callback = None):
        # allow single thread object
        if not hasattr(workers, '__len__'):
            workers = (workers,)

        for i in range(len(workers)):
            workers[i].start()

        #imgui.create_context()
        self.set_ui_scale(self.ui_scale)

        self._lock()
        impl = GlfwRenderer(self._window)
        self._unlock()
        
        self._pressed_keys = set()
        self._hit_keys = set()

        def on_key(window, key, scan, pressed, mods):
            if pressed:
                if key not in self._pressed_keys:
                    self._hit_keys.add(key)
                self._pressed_keys.add(key)
            else:
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key) # check seems to be needed over RDP sometimes
            if key != glfw.KEY_ESCAPE: # imgui erases text with escape (??)
                impl.keyboard_callback(window, key, scan, pressed, mods)

        glfw.set_key_callback(self._window, on_key)

        # For settings custom callbacks etc.
        if glfw_init_callback is not None:
            glfw_init_callback(self._window)

        while not (glfw.window_should_close(self._window) or self.keyhit(glfw.KEY_ESCAPE)):
            glfw.poll_events()
            impl.process_inputs()

            self._lock()
            imgui.get_io().display_size = glfw.get_framebuffer_size(self._window)
            imgui.new_frame()

            # Tero viewer:
            imgui.push_font(self._imgui_fonts[self._cur_font_size])
            self.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)
    
            loopfunc(self)

            for key in self._editables:
                self._editables[key].loop(self)

            imgui.pop_font()

            imgui.render()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self._window)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self._unlock()
        
        # Update size and pos
        if not self.fullscreen:
            self._width, self._height = glfw.get_framebuffer_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)

        with open(self._inifile, 'w') as file:
            file.write('{} {}\n'.format(self._width, self._height))
            file.write('{} {}\n'.format(*self.window_pos))
            file.write('{}\n'.format(glfw.get_window_attrib(self._window, glfw.MAXIMIZED)))
            file.write('{}\n'.format(self.ui_scale))
            file.write('{}\n'.format(int(self.fullscreen)))
            for k, e in self._editables.items():
                file.write(k+'\n')
                for code in (e.ui_code, e.run_code):
                    lines = code.split('\n')
                    file.write(str(len(lines))+'\n')
                    for line in lines:
                        file.write(line+'\n')

        self._lock()
        self.quit = True
        self._unlock()

        for i in range(len(workers)):
            workers[i].join()
            
        glfw.make_context_current(self._window)
        del self._images
        self._images = {}
        glfw.make_context_current(None)

        glfw.destroy_window(self._window)
        self.pop_context()

    def upload_image(self, name, data):
        if torch.is_tensor(data):
            return self.upload_image_torch(name, data)
        else:
            return self.upload_image_np(name, data)

    # Upload image from PyTorch tensor
    def upload_image_torch(self, name, tensor):
        assert isinstance(tensor, torch.Tensor)
        if self._lock():
            torch.cuda.synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_torch(tensor)
                self.pop_context()
            self._unlock()
    
    # Upload data from cuda pointer retrieved using custom TF op 
    def upload_image_TF_ptr(self, name, ptr, shape):
        if self._lock():
            torch.cuda.synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_ptr(ptr, shape)
                self.pop_context()
            self._unlock()

    def upload_image_np(self, name, data):
        assert isinstance(data, np.ndarray)
        if self._lock():
            torch.cuda.synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_np(data)
                self.pop_context()
            self._unlock()
