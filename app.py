import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, Slider, RectangleSelector
import os
from tkinter import Tk, filedialog
from matplotlib.colors import hsv_to_rgb

class ImageEditor:
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.filename = None
    
    def load_image(self, filepath=None):
        """Carga una imagen desde un archivo."""
        if filepath is None:
            root = Tk()
            root.withdraw()  # Ocultar la ventana principal
            filepath = filedialog.askopenfilename(
                title="Seleccionar imagen",
                filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
            )
            root.destroy()
            
            if not filepath:  # Si el usuario cancela
                return None
                
        self.filename = os.path.basename(filepath)
        img = mpimg.imread(filepath)
        
        # Asegurarse de que la imagen sea un array con valores entre 0-1
        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0
            
        # Si la imagen es grayscale, convertirla a RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        
        # Si la imagen tiene canal alpha (RGBA), convertirla a RGB
        if img.shape[2] == 4:
            # Extraer solo los canales RGB
            img = img[:, :, :3]
            
        self.current_image = img
        self.original_image = img.copy()
        return img
    
    def save_image(self, filepath=None):
        """Guarda la imagen actual en un archivo."""
        if self.current_image is None:
            print("No hay imagen para guardar.")
            return
            
        if filepath is None:
            root = Tk()
            root.withdraw()
            default_name = f"edited_{self.filename}" if self.filename else "edited_image.png"
            filepath = filedialog.asksaveasfilename(
                title="Guardar imagen como",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            
            if not filepath:  # Si el usuario cancela
                return
                
        # Convertir la imagen a uint8 para guardarla
        img_to_save = (self.current_image * 255).astype(np.uint8)
        plt.imsave(filepath, img_to_save)
        print(f"Imagen guardada en {filepath}")
    
    def display_image(self, image=None, title=None):
        """Muestra una imagen en una ventana de matplotlib."""
        if image is None:
            image = self.current_image
            
        if image is None:
            print("No hay imagen para mostrar.")
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Taller 8: Funciones de manejo de matrices e imágenes
    
    def invert_colors(self, image=None):
        """
        Invierte los colores de una imagen (función 3 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para invertir.")
            return None
            
        inverted = 1 - image
        self.current_image = inverted
        return inverted
    
    def extract_red_channel(self, image=None):
        """
        Extrae la capa roja de una imagen (función 4 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        red_channel = image.copy()
        red_channel[:, :, 1] = 0  # Poner capa verde en 0
        red_channel[:, :, 2] = 0  # Poner capa azul en 0
        
        self.current_image = red_channel
        return red_channel
    
    def extract_green_channel(self, image=None):
        """
        Extrae la capa verde de una imagen (función 5 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        green_channel = image.copy()
        green_channel[:, :, 0] = 0  # Poner capa roja en 0
        green_channel[:, :, 2] = 0  # Poner capa azul en 0
        
        self.current_image = green_channel
        return green_channel
    
    def extract_blue_channel(self, image=None):
        """
        Extrae la capa azul de una imagen (función 6 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        blue_channel = image.copy()
        blue_channel[:, :, 0] = 0  # Poner capa roja en 0
        blue_channel[:, :, 1] = 0  # Poner capa verde en 0
        
        self.current_image = blue_channel
        return blue_channel
    
    def convert_to_magenta(self, image=None):
        """
        Convierte una imagen a magenta (función 7 del Taller 8).
        Magenta = Rojo + Azul
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        magenta = image.copy()
        # Magenta es rojo + azul, así que eliminamos verde
        magenta[:, :, 1] = 0
        
        self.current_image = magenta
        return magenta
    
    def convert_to_cyan(self, image=None):
        """
        Convierte una imagen a cyan (función 8 del Taller 8).
        Cyan = Verde + Azul
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        cyan = image.copy()
        # Cyan es verde + azul, así que eliminamos rojo
        cyan[:, :, 0] = 0
        
        self.current_image = cyan
        return cyan
    
    def convert_to_yellow(self, image=None):
        """
        Convierte una imagen a amarillo (función 9 del Taller 8).
        Amarillo = Rojo + Verde
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        yellow = image.copy()
        # Amarillo es rojo + verde, así que eliminamos azul
        yellow[:, :, 2] = 0
        
        self.current_image = yellow
        return yellow
    
    def reconstruct_from_rgb(self, red_channel, green_channel, blue_channel):
        """
        Reconstruye una imagen a color a partir de sus capas RGB (función 10 del Taller 8).
        """
        if red_channel is None or green_channel is None or blue_channel is None:
            print("Faltan canales para reconstruir la imagen.")
            return None
            
        # Crear una imagen en blanco
        height, width = red_channel.shape[:2]
        reconstructed = np.zeros((height, width, 3))
        
        # Extraer solo el canal rojo de la primera imagen
        reconstructed[:, :, 0] = red_channel[:, :, 0]
        
        # Extraer solo el canal verde de la segunda imagen
        reconstructed[:, :, 1] = green_channel[:, :, 1]
        
        # Extraer solo el canal azul de la tercera imagen
        reconstructed[:, :, 2] = blue_channel[:, :, 2]
        
        self.current_image = reconstructed
        return reconstructed
    
    def merge_images_without_equalization(self, image1=None, image2=None):
        """
        Fusiona dos imágenes sin ecualizar (función 11 del Taller 8).
        La fusión se hace promediando los valores de cada píxel.
        """
        if image1 is None:
            image1 = self.current_image
            
        if image2 is None:
            # Cargar segunda imagen
            print("Seleccione la segunda imagen para fusionar:")
            image2 = self.load_image()
            self.current_image = image1  # Restaurar la imagen actual
            
        if image1 is None or image2 is None:
            print("Faltan imágenes para fusionar.")
            return None
            
        # Asegurar que las imágenes tengan las mismas dimensiones
        if image1.shape != image2.shape:
            # Redimensionar la segunda imagen para que coincida con la primera
            from skimage.transform import resize
            image2 = resize(image2, image1.shape, mode='reflect', anti_aliasing=True)
            
        # Fusionar las imágenes promediando sus valores
        merged = (image1 + image2) / 2
        
        self.current_image = merged
        return merged
    
    def merge_images_with_equalization(self, image1=None, image2=None):
        """
        Fusiona dos imágenes ecualizadas (función 12 del Taller 8).
        Primero ecualiza ambas imágenes y luego las fusiona.
        """
        if image1 is None:
            image1 = self.current_image
            
        if image2 is None:
            # Cargar segunda imagen
            print("Seleccione la segunda imagen para fusionar:")
            image2 = self.load_image()
            self.current_image = image1  # Restaurar la imagen actual
            
        if image1 is None or image2 is None:
            print("Faltan imágenes para fusionar.")
            return None
            
        # Asegurar que las imágenes tengan las mismas dimensiones
        if image1.shape != image2.shape:
            # Redimensionar la segunda imagen para que coincida con la primera
            from skimage.transform import resize
            image2 = resize(image2, image1.shape, mode='reflect', anti_aliasing=True)
            
        # Ecualizar ambas imágenes (factor 1.5 como ejemplo)
        equalized1 = self.equalize_image(image1, 1.5)
        equalized2 = self.equalize_image(image2, 1.5)
        
        # Fusionar las imágenes ecualizadas
        merged = (equalized1 + equalized2) / 2
        
        self.current_image = merged
        return merged
    
    def equalize_image(self, image=None, factor=1.5):
        """
        Ecualiza una imagen según un factor dado (función 13 del Taller 8).
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            print("No hay imagen para ecualizar.")
            return None
            
        # Implementación simple de ecualización con ajuste de contraste
        equalized = np.clip(image * factor, 0, 1)
        
        if image is self.current_image:
            self.current_image = equalized
        return equalized
    
    def apply_average_technique(self, image=None):
        """
        Aplica la técnica de promedio a una imagen (función 14 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        # Técnica de promedio: promediar los canales RGB para cada píxel
        average = image.copy()
        for i in range(3):  # Para cada canal
            average[:, :, i] = np.mean(image, axis=2)
            
        self.current_image = average
        return average
    
    def convert_to_grayscale_average(self, image=None):
        """
        Convierte una imagen a escala de grises usando la técnica de promedio (función 15 del Taller 8).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        # Calcular el promedio de los tres canales
        grayscale = np.mean(image, axis=2)
        
        # Convertir a formato RGB (aunque todos los canales tienen el mismo valor)
        grayscale_rgb = np.stack([grayscale, grayscale, grayscale], axis=2)
        
        self.current_image = grayscale_rgb
        return grayscale_rgb
    
    def convert_to_grayscale_luminosity(self, image=None):
        """
        Convierte una imagen a escala de grises usando la técnica de luminosidad (función 16 del Taller 8).
        Utiliza pesos: R*0.299 + G*0.587 + B*0.114
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        # Aplicar pesos a cada canal (basados en la percepción del ojo humano)
        grayscale = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        
        # Convertir a formato RGB (aunque todos los canales tienen el mismo valor)
        grayscale_rgb = np.stack([grayscale, grayscale, grayscale], axis=2)
        
        self.current_image = grayscale_rgb
        return grayscale_rgb
    
    def convert_to_grayscale_midgray(self, image=None):
        """
        Convierte una imagen a escala de grises usando la técnica de tonalidad (Midgray) (función 17 del Taller 8).
        Calcula (max + min) / 2 para cada píxel.
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para procesar.")
            return None
            
        # Calcular el valor medio entre el máximo y mínimo de los canales RGB
        max_val = np.max(image, axis=2)
        min_val = np.min(image, axis=2)
        midgray = (max_val + min_val) / 2
        
        # Convertir a formato RGB (aunque todos los canales tienen el mismo valor)
        midgray_rgb = np.stack([midgray, midgray, midgray], axis=2)
        
        self.current_image = midgray_rgb
        return midgray_rgb
    
    # Taller 9: Funciones de transformaciones
    
    def adjust_brightness(self, image=None, factor=1.2):
        """
        Ajusta el brillo de una imagen según un factor (función 1 del Taller 9).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para ajustar.")
            return None
            
        # Multiplicar todos los píxeles por el factor y recortar al rango [0, 1]
        adjusted = np.clip(image * factor, 0, 1)
        
        self.current_image = adjusted
        return adjusted
    
    def adjust_channel_brightness(self, image=None, channel=0, factor=1.2):
        """
        Ajusta el brillo de un canal específico según un factor (función 2 del Taller 9).
        channel: 0=Rojo, 1=Verde, 2=Azul
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para ajustar.")
            return None
            
        if channel not in [0, 1, 2]:
            print("Canal inválido. Debe ser 0 (Rojo), 1 (Verde) o 2 (Azul).")
            return None
            
        adjusted = image.copy()
        adjusted[:, :, channel] = np.clip(image[:, :, channel] * factor, 0, 1)
        
        self.current_image = adjusted
        return adjusted
    
    def adjust_contrast(self, image=None, factor=1.5):
        """
        Ajusta el contraste de una imagen según un factor (función 3 del Taller 9).
        Factor > 1 aumenta el contraste, factor < 1 disminuye el contraste.
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para ajustar.")
            return None
            
        # Calcular el valor medio de la imagen
        mean = np.mean(image)
        
        # Aplicar la fórmula: nuevo = (original - mean) * factor + mean
        adjusted = np.clip((image - mean) * factor + mean, 0, 1)
        
        self.current_image = adjusted
        return adjusted
    
    def decrease_contrast(self, image=None, factor=0.7):
        """
        Disminuye el contraste de una imagen según un factor.
        """
        # Simplemente llamamos a adjust_contrast con un factor < 1
        return self.adjust_contrast(image, factor)
    
    def zoom_image(self, image=None, x_start=None, y_start=None, x_end=None, y_end=None):
        """
        Realiza zoom en una parte de la imagen (función 4 del Taller 9).
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            print("No hay imagen para aplicar zoom.")
            return None
            
        # Si no se proporcionan coordenadas, permitir al usuario seleccionar un área
        if x_start is None or y_start is None or x_end is None or y_end is None:
            print("Seleccione un área para hacer zoom (arrastre el ratón).")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_title("Seleccione área para zoom")
            
            # Variables para almacenar la selección
            coords = {}
            
            def onselect(eclick, erelease):
                x1, y1 = int(eclick.xdata), int(eclick.ydata)
                x2, y2 = int(erelease.xdata), int(erelease.ydata)
                coords['x1'] = min(x1, x2)
                coords['y1'] = min(y1, y2)
                coords['x2'] = max(x1, x2)
                coords['y2'] = max(y1, y2)
                plt.close()
                
            rs = RectangleSelector(
                 ax, onselect,
                 props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True),
                 interactive=True
            )
            plt.show()
            
            # Usar las coordenadas seleccionadas
            if not coords:  # Si el usuario cerró la ventana sin seleccionar
                return None
                
            x_start, y_start = coords['x1'], coords['y1']
            x_end, y_end = coords['x2'], coords['y2']
            
        # Verificar que las coordenadas estén dentro de los límites de la imagen
        height, width = image.shape[:2]
        x_start = max(0, min(x_start, width - 1))
        y_start = max(0, min(y_start, height - 1))
        x_end = max(0, min(x_end, width))
        y_end = max(0, min(y_end, height))
        
        # Extraer la región seleccionada
        zoomed = image[y_start:y_end, x_start:x_end, :]
        
        self.current_image = zoomed
        return zoomed
    
    def binarize_image(self, image=None, threshold=0.5):
        """
        Binariza una imagen según un umbral (función 5 del Taller 9).
        """
        if image is None:
            image = self.original_image.copy()  # Usar copia de la original
            
        if image is None:
            print("No hay imagen para binarizar.")
            return None
            
        # Primero convertimos a escala de grises usando la técnica de luminosidad
        grayscale = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        
        # Aplicar umbral
        binary = (grayscale > threshold).astype(float)
        
        # Convertir a formato RGB (aunque todos los canales tienen el mismo valor)
        binary_rgb = np.stack([binary, binary, binary], axis=2)
        
        self.current_image = binary_rgb
        return binary_rgb
    
    def rotate_image(self, image=None, angle=90):
        """
        Rota una imagen (función 6 del Taller 9).
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            print("No hay imagen para rotar.")
            return None
            
        # Implementar rotación usando numpy
        if angle == 90:
            rotated = np.rot90(image, k=1)
        elif angle == 180:
            rotated = np.rot90(image, k=2)
        elif angle == 270:
            rotated = np.rot90(image, k=3)
        else:
            # Para ángulos arbitrarios, usamos matplotlib
            from matplotlib import transforms
            
            # Crear figura temporal para la rotación
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            
            # Aplicar la transformación de rotación
            transform = transforms.Affine2D().rotate_deg(angle) + ax.transData
            ax.images[0].set_transform(transform)
            
            # Ajustar los límites
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            
            # Capturar la imagen rotada
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            rotated = data / 255.0  # Normalizar a [0, 1]
            
            plt.close(fig)
        
        self.current_image = rotated
        return rotated
    
    def compute_histogram(self, image=None):
        """
        Calcula y muestra el histograma de cada canal de la imagen (función 7 del Taller 9).
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            print("No hay imagen para calcular el histograma.")
            return None
            
        # Crear una figura con subplots para cada canal
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        colors = ['red', 'green', 'blue']
        
        for i, color in enumerate(colors):
            # Calcular histograma del canal
            hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=(0, 1))
            
            # Mostrar histograma
            axs[i].bar(bins[:-1], hist, width=1/256, color=color, alpha=0.7)
            axs[i].set_title(f'Histograma del canal {color}')
            axs[i].set_xlabel('Valor del píxel')
            axs[i].set_ylabel('Frecuencia')
            axs[i].set_xlim(0, 1)
            
        plt.tight_layout()
        plt.show()
        
        return hist, bins

    def create_ui(self):
        """
        Crea una interfaz gráfica simple para el editor de imágenes.
        """
        # Cargar una imagen para empezar
        if self.current_image is None:
            self.load_image()
            
        if self.current_image is None:
            print("No se cargó ninguna imagen. Saliendo.")
            return
            
        # Crear la interfaz gráfica
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.canvas.manager.set_window_title('Editor de Imágenes')
        
        img_display = ax.imshow(self.current_image)
        ax.set_title("Editor de Imágenes")
        ax.axis('off')
        
        # Crear botones para las diferentes funciones
        buttons_height = 0.05
        buttons_width = 0.1
        buttons_spacing = 0.01
        
        # Primera fila de botones
        button_positions = [
            ('Cargar', 0.05),
            ('Guardar', 0.16),
            ('Original', 0.27),
            ('Invertir', 0.38),
            ('Rojo', 0.49),
            ('Verde', 0.60),
            ('Azul', 0.71),
            ('Gris Avg', 0.82)
        ]
        
        buttons = []
        for label, position in button_positions:
            ax_button = plt.axes([position, 0.01, buttons_width, buttons_height])
            button = Button(ax_button, label)
            buttons.append(button)
            
        # Segunda fila de botones
        button_positions2 = [
            ('Magenta', 0.05),
            ('Cyan', 0.16),
            ('Amarillo', 0.27),
            ('Brillo +', 0.38),
            ('Brillo -', 0.49),
            ('Contraste +', 0.60),
            ('Binarizar', 0.71),
            ('Rotar', 0.82)
        ]
        
        buttons2 = []
        for label, position in button_positions2:
            ax_button = plt.axes([position, 0.07, buttons_width, buttons_height])
            button = Button(ax_button, label)
            buttons2.append(button)
            
        # Tercera fila de botones
        button_positions3 = [
            ('Gris Lum', 0.05),
            ('Gris Mid', 0.16),
            ('Histograma', 0.27),
            ('Zoom', 0.38),
            ('Fusionar', 0.49),
            ('Ecualizar', 0.60),
            ('Promedio', 0.71),
            ('Salir', 0.82)
        ]
        
        buttons3 = []
        for label, position in button_positions3:
            ax_button = plt.axes([position, 0.13, buttons_width, buttons_height])
            button = Button(ax_button, label)
            buttons3.append(button)
            
        # Funciones de callback para los botones
        def on_load(event):
            self.load_image()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_save(event):
            self.save_image()
            
        def on_original(event):
            self.current_image = self.original_image.copy()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_invert(event):
            self.invert_colors()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_red(event):
            self.current_image = self.extract_red_channel()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_green(event):
            self.current_image = self.extract_green_channel()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_blue(event):
            self.current_image = self.extract_blue_channel()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_gray_avg(event):
            self.convert_to_grayscale_average()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_magenta(event):
            self.convert_to_magenta()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_cyan(event):
            self.convert_to_cyan()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_yellow(event):
            self.convert_to_yellow()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_brightness_up(event):
            self.adjust_brightness(factor=1.2)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_brightness_down(event):
            self.adjust_brightness(factor=0.8)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_contrast_up(event):
            self.adjust_contrast(factor=1.5)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_binarize(event):
            self.binarize_image(threshold=0.5)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_rotate(event):
            self.rotate_image(angle=90)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_gray_lum(event):
            self.convert_to_grayscale_luminosity()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_gray_mid(event):
            self.convert_to_grayscale_midgray()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_histogram(event):
            self.compute_histogram()
            
        def on_zoom(event):
            plt.close(fig)  # Cerrar ventana actual
            zoomed = self.zoom_image()
            if zoomed is not None:
                self.create_ui()  # Reabrir la interfaz con la imagen ampliada
            
        def on_merge(event):
            plt.close(fig)  # Cerrar ventana actual
            merged = self.merge_images_without_equalization()
            if merged is not None:
                self.create_ui()  # Reabrir la interfaz con la imagen fusionada
            
        def on_equalize(event):
            self.equalize_image(factor=1.5)
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_average(event):
            self.apply_average_technique()
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
            
        def on_exit(event):
            plt.close(fig)
            
        # Asignar funciones a los botones
        buttons[0].on_clicked(on_load)
        buttons[1].on_clicked(on_save)
        buttons[2].on_clicked(on_original)
        buttons[3].on_clicked(on_invert)
        buttons[4].on_clicked(on_red)
        buttons[5].on_clicked(on_green)
        buttons[6].on_clicked(on_blue)
        buttons[7].on_clicked(on_gray_avg)
        
        buttons2[0].on_clicked(on_magenta)
        buttons2[1].on_clicked(on_cyan)
        buttons2[2].on_clicked(on_yellow)
        buttons2[3].on_clicked(on_brightness_up)
        buttons2[4].on_clicked(on_brightness_down)
        buttons2[5].on_clicked(on_contrast_up)
        buttons2[6].on_clicked(on_binarize)
        buttons2[7].on_clicked(on_rotate)
        
        buttons3[0].on_clicked(on_gray_lum)
        buttons3[1].on_clicked(on_gray_mid)
        buttons3[2].on_clicked(on_histogram)
        buttons3[3].on_clicked(on_zoom)
        buttons3[4].on_clicked(on_merge)
        buttons3[5].on_clicked(on_equalize)
        buttons3[6].on_clicked(on_average)
        buttons3[7].on_clicked(on_exit)
        
        plt.subplots_adjust(bottom=0.2)
        plt.show()

# Código para ejecutar la aplicación
if __name__ == "__main__":
    editor = ImageEditor()
    editor.create_ui()