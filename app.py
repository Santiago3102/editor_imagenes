import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, Slider, RectangleSelector, TextBox
import os
from tkinter import Tk, filedialog
from matplotlib.colors import hsv_to_rgb

class ImageEditor:
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.filename = None
        self.second_image = None  # Para almacenar la segunda imagen en fusión
    
    def load_image(self, filepath=None):
        """Carga una imagen desde un archivo."""
        try:
            root = Tk()
            root.withdraw()  # Ocultar la ventana principal
            if filepath is None:
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
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return None
    
    def save_image(self, filepath=None):
        """Guarda la imagen actual en un archivo."""
        if self.current_image is None:
            print("No hay imagen para guardar.")
            return
        
        try:    
            root = Tk()
            root.withdraw()
            default_name = f"edited_{self.filename}" if self.filename else "edited_image.png"
            if filepath is None:
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
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")
    
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
        Fusiona dos imágenes sin ecualizar.
        La fusión se hace promediando los valores de cada píxel.
        """
        if image1 is None:
            image1 = self.current_image
            
        if image2 is None and self.second_image is None:
            # Guardar la imagen actual temporalmente
            temp_image = self.current_image
            
            # Abrir selector de archivos sin interferir con el bucle de eventos de matplotlib
            import threading
            import os
            from tkinter import Tk, filedialog
            
            # Variable para almacenar la ruta del archivo seleccionado
            selected_filepath = [None]
            
            def file_dialog():
                root = Tk()
                root.withdraw()
                # Asegurar que la ventana aparezca por encima
                root.attributes('-topmost', True)
                # Seleccionar archivo
                filepath = filedialog.askopenfilename(
                    parent=root,
                    title="Seleccionar segunda imagen",
                    filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
                )
                selected_filepath[0] = filepath
                root.destroy()
            
            # Ejecutar diálogo en un hilo separado
            dialog_thread = threading.Thread(target=file_dialog)
            dialog_thread.start()
            dialog_thread.join()  # Esperar a que se complete
            
            # Procesar la imagen seleccionada
            filepath = selected_filepath[0]
            if filepath:
                try:
                    import matplotlib.image as mpimg
                    import numpy as np
                    
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
                        
                    self.second_image = img
                    
                except Exception as e:
                    print(f"Error al cargar la segunda imagen: {e}")
                    self.current_image = temp_image
                    return None
            else:
                print("No se seleccionó ninguna imagen.")
                return None
                
            # Restaurar la imagen actual
            image2 = self.second_image
            
        elif image2 is None:
            image2 = self.second_image
            
        if image1 is None or image2 is None:
            print("Faltan imágenes para fusionar.")
            return None
            
        # Asegurar que las imágenes tengan las mismas dimensiones
        if image1.shape != image2.shape:
            # Redimensionar la segunda imagen para que coincida con la primera
            try:
                from skimage.transform import resize
                image2 = resize(image2, image1.shape, mode='reflect', anti_aliasing=True)
            except ImportError:
                # Alternativa si no está disponible skimage
                # Implementación básica basada en tu código de fusión alternativo
                altura, ancho = image1.shape[:2]
                temp_image2 = np.zeros_like(image1)
                
                # Redimensionar manualmente cada canal
                for i in range(3):  # Para cada canal RGB
                    img2_channel = image2[:,:,i]
                    # Redimensionar utilizando interpolación
                    y_orig, x_orig = np.mgrid[0:image2.shape[0]-1:complex(0,altura), 
                                            0:image2.shape[1]-1:complex(0,ancho)]
                    y_new = np.linspace(0, image2.shape[0]-1, altura)
                    x_new = np.linspace(0, image2.shape[1]-1, ancho)
                    
                    from scipy import interpolate
                    f = interpolate.interp2d(np.arange(image2.shape[1]), np.arange(image2.shape[0]), img2_channel)
                    temp_image2[:,:,i] = f(x_new, y_new)
                
                image2 = temp_image2
            
        # Fusionar las imágenes promediando sus valores
        merged = (image1 + image2) / 2
        
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
            
        # Implementación correcta de ajuste de brillo: 
        # Añadir un valor constante a todos los píxeles
        # factor > 0 aumenta brillo, factor < 0 disminuye brillo
        adjusted = np.clip(image + factor - 1.0, 0, 1)
        
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
            
        # Implementar rotación usando numpy para los ángulos comunes
        if angle == 90:
            rotated = np.rot90(image, k=1)
        elif angle == 180:
            rotated = np.rot90(image, k=2)
        elif angle == 270 or angle == -90:
            rotated = np.rot90(image, k=3)
        else:
            # Para ángulos arbitrarios, usamos scipy con mejor manejo de memoria
            from scipy import ndimage
            
            # Reducir resolución temporalmente para ángulos arbitrarios 
            # si la imagen es grande (mejora rendimiento)
            h, w = image.shape[:2]
            if h > 1000 or w > 1000:
                from skimage.transform import resize
                # Reducir tamaño para procesamiento
                scale_factor = 1000 / max(h, w)
                small_img = resize(image, (int(h*scale_factor), int(w*scale_factor)), 
                                mode='reflect', anti_aliasing=True)
                
                # Rotar la imagen reducida
                rotated = ndimage.rotate(small_img, angle, reshape=True, mode='reflect')
                
                # Volver a escalar al tamaño original si es necesario
                if angle % 90 != 0:  # Solo para ángulos que cambian las dimensiones
                    new_h, new_w = rotated.shape[:2]
                    # Calcular el factor de escala para volver al tamaño aproximado original
                    target_size = max(h, w)
                    scale_back = target_size / max(new_h, new_w)
                    rotated = resize(rotated, (int(new_h*scale_back), int(new_w*scale_back)), 
                                    mode='reflect', anti_aliasing=True)
            else:
                # Si la imagen no es tan grande, rotar directamente
                rotated = ndimage.rotate(image, angle, reshape=True, mode='reflect')
                
            # Recortar valores fuera de rango
            rotated = np.clip(rotated, 0, 1)
        
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
        Crea una interfaz gráfica mejorada para el editor de imágenes.
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
        
        # Reorganizar la interfaz para mostrar controles en los laterales y abajo
        # Añadir un espacio para sliders y controles a la derecha de la imagen
        
        # Definir áreas y disposición de la interfaz
        fig.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.95)
        
        # Área para botones principales en la parte inferior
        button_width = 0.07
        button_height = 0.04
        button_spacing = 0.01
        
        # Primera fila de botones (abajo)
        button_row1_y = 0.05
        buttons_row1 = [
            ('Cargar', 0.05),
            ('Guardar', 0.13),
            ('Original', 0.21),
            ('Invertir', 0.29),
            ('Rojo', 0.37),
            ('Verde', 0.45),
            ('Azul', 0.53),
            ('Gris Avg', 0.61),
            ('Histograma', 0.69),
            ('Salir', 0.77)
        ]
        
        buttons1 = []
        for label, position in buttons_row1:
            ax_button = plt.axes([position, button_row1_y, button_width, button_height])
            button = Button(ax_button, label)
            buttons1.append(button)
            
        # Segunda fila de botones (abajo)
        button_row2_y = 0.10
        buttons_row2 = [
            ('Magenta', 0.05),
            ('Cyan', 0.13),
            ('Amarillo', 0.21),
            ('Gris Lum', 0.29),
            ('Gris Mid', 0.37),
            ('Zoom', 0.45),
            ('Fusionar', 0.53),
            ('Ecualizar', 0.61),
            ('Promedio', 0.69),
            ('Binarizar', 0.77)
        ]
        
        buttons2 = []
        for label, position in buttons_row2:
            ax_button = plt.axes([position, button_row2_y, button_width, button_height])
            button = Button(ax_button, label)
            buttons2.append(button)
            
        # Controles a la derecha
        right_panel_x = 0.87
        control_width = 0.1
        control_height = 0.03
        
        # Slider para brillo
        brightness_slider_ax = plt.axes([right_panel_x, 0.8, control_width, control_height])
        brightness_slider = Slider(
            brightness_slider_ax, 'Brillo', 
            0.0, 2.0, 
            valinit=1.0, 
            valstep=0.1
        )
        
        # Slider para contraste
        contrast_slider_ax = plt.axes([right_panel_x, 0.7, control_width, control_height])
        contrast_slider = Slider(
            contrast_slider_ax, 'Contraste', 
            0.1, 3.0, 
            valinit=1.0, 
            valstep=0.1
        )
        
        # Slider para umbral de binarización (AGREGADO)
        binarize_slider_ax = plt.axes([right_panel_x, 0.6, control_width, control_height])
        binarize_slider = Slider(
            binarize_slider_ax, 'Umbral', 
            0.0, 1.0, 
            valinit=0.5, 
            valstep=0.05
        )
        
        # Botones para aplicar brillo y contraste
        apply_brightness_ax = plt.axes([right_panel_x, 0.5, control_width, control_height])
        apply_brightness_button = Button(apply_brightness_ax, 'Aplicar Brillo')
        
        apply_contrast_ax = plt.axes([right_panel_x, 0.4, control_width, control_height])
        apply_contrast_button = Button(apply_contrast_ax, 'Aplicar Contraste')
        
        # Botones para rotación
        rotate_90_ax = plt.axes([right_panel_x, 0.3, control_width, control_height])
        rotate_90_button = Button(rotate_90_ax, 'Rotar 90°')
        
        rotate_180_ax = plt.axes([right_panel_x, 0.25, control_width, control_height])
        rotate_180_button = Button(rotate_180_ax, 'Rotar 180°')
        
        rotate_270_ax = plt.axes([right_panel_x, 0.2, control_width, control_height])
        rotate_270_button = Button(rotate_270_ax, 'Rotar 270°')
        
        # Textbox para rotación personalizada
        rotate_custom_ax = plt.axes([right_panel_x, 0.15, control_width, control_height])
        rotate_custom_textbox = TextBox(rotate_custom_ax, 'Ángulo:', initial='0')
        
        rotate_apply_ax = plt.axes([right_panel_x, 0.1, control_width, control_height])
        rotate_apply_button = Button(rotate_apply_ax, 'Aplicar Rot.')
        
        # Definir callbacks para los botones y sliders
        def update_display():
            """Actualiza la imagen mostrada en la interfaz."""
            img_display.set_data(self.current_image)
            fig.canvas.draw_idle()
        
        # Callbacks para la primera fila de botones
        def on_load_button(event):
            self.load_image()
            update_display()
        
        def on_save_button(event):
            self.save_image()
        
        def on_original_button(event):
            self.current_image = self.original_image.copy()
            update_display()
        
        def on_invert_button(event):
            self.invert_colors()
            update_display()
        
        def on_red_button(event):
            self.extract_red_channel()
            update_display()
        
        def on_green_button(event):
            self.extract_green_channel()
            update_display()
        
        def on_blue_button(event):
            self.extract_blue_channel()
            update_display()
        
        def on_gray_avg_button(event):
            self.convert_to_grayscale_average()
            update_display()
        
        def on_histogram_button(event):
            self.compute_histogram()
        
        def on_exit_button(event):
            plt.close(fig)
        
        # Callbacks para la segunda fila de botones
        def on_magenta_button(event):
            self.convert_to_magenta()
            update_display()
        
        def on_cyan_button(event):
            self.convert_to_cyan()
            update_display()
        
        def on_yellow_button(event):
            self.convert_to_yellow()
            update_display()
        
        def on_gray_lum_button(event):
            self.convert_to_grayscale_luminosity()
            update_display()
        
        def on_gray_mid_button(event):
            self.convert_to_grayscale_midgray()
            update_display()
        
        def on_zoom_button(event):
            self.zoom_image()
            update_display()
        
        def on_merge_button(event):
            self.merge_images_without_equalization()
            update_display()
        
        def on_equalize_button(event):
            self.equalize_image(factor=1.5)
            update_display()
        
        def on_average_button(event):
            self.apply_average_technique()
            update_display()
        
        def on_binarize_button(event):
            threshold = binarize_slider.val
            self.binarize_image(threshold=threshold)
            update_display()
        
        # Callbacks para los sliders y botones de control
        def on_apply_brightness(event):
            factor = brightness_slider.val
            self.adjust_brightness(factor=factor)
            update_display()
        
        def on_apply_contrast(event):
            factor = contrast_slider.val
            self.adjust_contrast(factor=factor)
            update_display()
        
        def on_rotate_90(event):
            self.rotate_image(angle=90)
            update_display()
        
        def on_rotate_180(event):
            self.rotate_image(angle=180)
            update_display()
        
        def on_rotate_270(event):
            self.rotate_image(angle=270)
            update_display()
        
        def on_rotate_custom(event):
            try:
                angle = float(rotate_custom_textbox.text)
                self.rotate_image(angle=angle)
                update_display()
            except ValueError:
                print("Por favor ingrese un valor numérico para el ángulo.")
        
        # Conectar callbacks a los botones de la primera fila
        buttons1[0].on_clicked(on_load_button)
        buttons1[1].on_clicked(on_save_button)
        buttons1[2].on_clicked(on_original_button)
        buttons1[3].on_clicked(on_invert_button)
        buttons1[4].on_clicked(on_red_button)
        buttons1[5].on_clicked(on_green_button)
        buttons1[6].on_clicked(on_blue_button)
        buttons1[7].on_clicked(on_gray_avg_button)
        buttons1[8].on_clicked(on_histogram_button)
        buttons1[9].on_clicked(on_exit_button)
        
        # Conectar callbacks a los botones de la segunda fila
        buttons2[0].on_clicked(on_magenta_button)
        buttons2[1].on_clicked(on_cyan_button)
        buttons2[2].on_clicked(on_yellow_button)
        buttons2[3].on_clicked(on_gray_lum_button)
        buttons2[4].on_clicked(on_gray_mid_button)
        buttons2[5].on_clicked(on_zoom_button)
        buttons2[6].on_clicked(on_merge_button)
        buttons2[7].on_clicked(on_equalize_button)
        buttons2[8].on_clicked(on_average_button)
        buttons2[9].on_clicked(on_binarize_button)
        
        # Conectar callbacks a los controles de la derecha
        apply_brightness_button.on_clicked(on_apply_brightness)
        apply_contrast_button.on_clicked(on_apply_contrast)
        rotate_90_button.on_clicked(on_rotate_90)
        rotate_180_button.on_clicked(on_rotate_180)
        rotate_270_button.on_clicked(on_rotate_270)
        rotate_apply_button.on_clicked(on_rotate_custom)
        
        plt.show()

# Para usar la clase ImageEditor, necesitamos crear una instancia y llamar a create_ui()
if __name__ == "__main__":
    editor = ImageEditor()
    editor.create_ui()