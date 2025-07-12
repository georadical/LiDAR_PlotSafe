#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 2025 LiDAR PlotSafe Project. All rights reserved.

Launcher GUI for LiDAR PlotSafe processing pipeline.

This module provides a simple graphical interface to select point cloud files and run the processing pipeline.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback

# Import LiDAR PlotSafe modules
from src import io

# Path configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LiDARPlotSafeLauncher(tk.Tk):
    """
    Mini graphical interface for the LiDAR PlotSafe pipeline.
    
    Allows selecting input files and navigating through different stages of the analysis process, showing progress.
    """
    
    # Define color scheme as class variables
    # Definir esquema de colores como variables de clase
    HEADER_COLOR = "#2c3e50"  # Dark blue-gray for header
    BACKGROUND_COLOR = "#f5f5f5"  # Light gray-white for background
    CONTENT_COLOR = "#ecf0f1"  # Light gray-white for content area
    ACCENT_COLOR = "#3498db"  # Blue for accents and highlights
    TEXT_COLOR = "#333333"  # Dark gray for text
    BUTTON_COLOR = "#ecf0f1"  # Very light gray for buttons
    
    def __init__(self):
        """
        Initializes the GUI application.
        
        Inicializa la aplicación GUI.
        """
        super().__init__()
        
        # Main window configuration
        # Configuración de la ventana principal
        self.title("LiDAR PlotSafe")
        self.geometry("700x500")  # Reduced height
        self.resizable(True, True)
        self.configure(bg=self.BACKGROUND_COLOR)  # Set window background
        
        # Set icon if available
        # Establecer ícono si está disponible
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "../..", "resources", "icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception:
            pass  # Icon not critical, continue without it
        
        # Initialize styles once for all widgets
        # Inicializar estilos una vez para todos los widgets
        self._initialize_styles()
        
        # Initialize controller variables
        # Inicializar variables del controlador
        self._initialize_variables()
        
        # Single main frame with proper styling - no nested containers
        # Un solo marco principal con estilo adecuado - sin contenedores anidados
        self.main_frame = ttk.Frame(self, style="Content.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add main title at the top with additional padding
        # Añadir título principal en la parte superior con padding adicional
        title_label = ttk.Label(
            self.main_frame,
            text="LiDAR PlotSafe Tool",
            font=("Arial", 16, "bold"),
            foreground=self.TEXT_COLOR,
            background=self.CONTENT_COLOR
        )
        # Increased padding above and below the title to create a more comfortable space around it
        # Aumentado el padding arriba y abajo del título para crear un espacio más cómodo alrededor
        title_label.pack(pady=(15, 20))  
        
        # Content frame for different views
        # Marco de contenido para diferentes vistas
        content_frame = ttk.Frame(self.main_frame, style="Content.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for different views
        # Crear marcos para diferentes vistas
        self.frames = {}
        for F in (LoadFrame, ParametersFrame):
            frame = F(content_frame, self)
            self.frames[F.__name__] = frame
            frame.pack(fill=tk.BOTH, expand=True)
            frame.pack_forget()  # Hide initially
        
        # Show initial frame
        # Mostrar el marco inicial
        self.show_frame("LoadFrame")

    def _initialize_styles(self):
        """
        Initialize ttk styles for the application.
        
        Inicializa los estilos ttk para la aplicación.
        """
        style = ttk.Style()
        
        # Configure basic styles using our color scheme
        # Configurar estilos básicos usando nuestro esquema de colores
        style.configure("TFrame", background=self.CONTENT_COLOR)
        style.configure("Content.TFrame", background=self.CONTENT_COLOR)
        style.configure("TLabel", background=self.CONTENT_COLOR, foreground=self.TEXT_COLOR)
        style.configure("TButton", background=self.BUTTON_COLOR)
        style.configure("TLabelframe", background=self.CONTENT_COLOR)
        style.configure("TLabelframe.Label", background=self.CONTENT_COLOR, foreground=self.TEXT_COLOR)
        
        # Style for progress bar
        # Estilo para la barra de progreso
        style.configure("TProgressbar", thickness=10)
        
    def _initialize_variables(self):
        """
        Initializes the controller variables.
        
        Inicializa las variables del controlador.
        """
        # Variables to store paths and parameters
        # Variables para almacenar rutas y parámetros
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.project_dir = tk.StringVar()
        self.raw_file_link = tk.StringVar()
        
        # Set default values
        # Establecer valores por defecto
        
        # Project structure status
        # Estado de la estructura del proyecto
        self.project_structure_created = False
        
        # Processing state
        self.processing = False
        self.current_step = 0
        self.total_steps = 0
    
    def show_frame(self, frame_name):
        """
        Shows the specified frame.
        
        Muestra el marco especificado.
        
        Args:
            frame_name: Name of the frame to display.
        """
        frame = self.frames[frame_name]
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Hide other frames
        for other_frame_name, other_frame in self.frames.items():
            if other_frame_name != frame_name:
                other_frame.pack_forget()
        
    def update_status(self, message, success=None):
        """
        Updates the status message in the current frame.
        
        Updates the status message in the current frame.
        
        Args:
            message: Message to display.
            success: True if the operation was successful, False if it failed, None for neutral state.
        """
        current_frame = None
        for frame_name, frame in self.frames.items():
            if frame.winfo_ismapped():
                current_frame = frame
                break
        
        if current_frame and hasattr(current_frame, "update_status"):
            current_frame.update_status(message, success)
    
    def update_progress(self, value, message=None):
        """
        Updates the progress bar in the current frame.
        
        Updates the progress bar in the current frame.
        
        Args:
            value: Value of the progress bar (0-100).
            message: Optional message to display.
        """
        current_frame = None
        for frame_name, frame in self.frames.items():
            if frame.winfo_ismapped():
                current_frame = frame
                break
        
        if current_frame and hasattr(current_frame, "update_progress"):
            current_frame.update_progress(value)
            
        if message and hasattr(current_frame, "update_status"):
            current_frame.update_status(message)


class LoadFrame(ttk.Frame):
    """
    Frame for loading point cloud files.
    
    Marco para cargar archivos de nubes de puntos.
    """
    
    def __init__(self, parent, controller):
        """
        Initializes the loading frame.
        
        Inicializa el marco de carga.
        
        Args:
            parent: Parent widget.
            controller: Main application controller.
        """
        super().__init__(parent)
        self.controller = controller
        
        # Create a style for consistent widget appearance
        # Crear un estilo para apariencia consistente de widgets
        style = ttk.Style()
        style.configure("TFrame", background=self.controller.CONTENT_COLOR)
        style.configure("TButton", background=self.controller.BUTTON_COLOR, font=("Arial", 10))
        style.configure("TLabel", background=self.controller.CONTENT_COLOR, font=("Arial", 10), foreground=self.controller.TEXT_COLOR)
        
        # Create the content frame that will contain all views
        # Crear el marco de contenido que contendrá todas las vistas
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create main content with proper spacing
        # Crear contenido principal con espaciado adecuado
        content_area = ttk.Frame(self.content_frame)
        content_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Buttons frame with compact layout
        # Marco de botones con diseño compacto
        buttons_frame = ttk.Frame(content_area)
        buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Set project button
        # Botón para establecer proyecto
        self.set_project_button = ttk.Button(
            buttons_frame, 
            text="Set Project Directory",
            command=self._set_project_directory,
            width=20
        )
        self.set_project_button.pack(pady=(0, 5))
        
        # Load file button
        self.load_button = ttk.Button(
            buttons_frame, 
            text="Load point cloud file",
            command=self._load_point_cloud,
            width=20,
            state="disabled"  # Initially disabled
        )
        self.load_button.pack(pady=0)
        
        # Progress bar and percentage in same frame
        # Barra de progreso y porcentaje en el mismo frame
        progress_frame = ttk.Frame(content_area)
        progress_frame.pack(fill=tk.X, padx=50, pady=(20, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=450,  # Reduced for space
            mode="determinate"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Percentage label positioned to the right of progress bar
        # Etiqueta de porcentaje posicionada a la derecha de la barra de progreso
        self.percentage_label = ttk.Label(
            progress_frame,  # Important change: now in progress_frame, not in self
            text="100%",
            font=("Arial", 10, "bold"),
            foreground="green",
            width=5
        )
        self.percentage_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.percentage_label.pack_forget()  # Initially hidden
        
        # Status message (initially empty)
        # Mensaje de estado (inicialmente vacío)
        self.status_label = ttk.Label(
            self, 
            text="",
            font=("Arial", 11),
            foreground=self.controller.ACCENT_COLOR,
            background=self.controller.CONTENT_COLOR,
            borderwidth=1,
            relief="solid",
            padding=5,
            anchor="center",  
            justify="center"  
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=(5, 10))
        
        # File info frame with minimized padding
        # Marco de información del archivo con padding minimizado
        self.file_info_frame = ttk.LabelFrame(
            content_area,
            text="Point Cloud Summary",
            padding=5
        )
        self.file_info_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.file_info_frame.pack_forget()  # Initially hidden
        
        # File information labels
        # Etiquetas de información del archivo
        self.filename_label = ttk.Label(
            self.file_info_frame,
            text="Filename: ",
            justify="left",
            anchor="w"
        )
        self.filename_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        self.points_label = ttk.Label(
            self.file_info_frame,
            text="Points: ",
            justify="left",
            anchor="w"
        )
        self.points_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        self.dimensions_label = ttk.Label(
            self.file_info_frame,
            text="Dimensions: ",
            justify="left",
            anchor="w"
        )
        self.dimensions_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        self.density_label = ttk.Label(
            self.file_info_frame,
            text="Density: ",
            justify="left",
            anchor="w"
        )
        self.density_label.grid(row=3, column=0, sticky="w", padx=5, pady=1)
        
        # Bottom buttons frame
        button_frame = ttk.Frame(content_area)
        button_frame.pack(fill=tk.X, padx=50, pady=20)
        
        # Left side buttons
        # Botones del lado izquierdo
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT, fill=tk.X)
        
        # Preview button
        self.preview_button = ttk.Button(
            left_buttons, 
            text="Preview point cloud",
            command=self._preview_point_cloud,
            state="disabled"
        )
        self.preview_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Remove file button
        # Botón para quitar el archivo actual
        self.remove_button = ttk.Button(
            left_buttons, 
            text="Remove file",
            command=self._remove_file,
            state="disabled"
        )
        self.remove_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side - Next button
        # Lado derecho - Botón siguiente
        self.next_button = ttk.Button(
            button_frame, 
            text="Next ->",
            command=self._next_step,
            state="disabled"
        )
        self.next_button.pack(side=tk.RIGHT)
    
    def _load_point_cloud(self):
        """
        Opens a dialog to select the point cloud file.
        
        Opens a dialog to select the point cloud file.
        """
        # Define supported file types based on io module
        extensions = io.get_supported_extensions()
        filetypes = [
            ("LiDAR Files", "*." + " *.".join(extensions)),
            *[(f"{ext.upper()} Files", f"*.{ext}") for ext in extensions],
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select point cloud file",
            filetypes=filetypes
        )
        
        if filename:
            # Save the file path
            self.controller.input_path.set(filename)
            
            # Start loading process in a separate thread
            self.load_button["state"] = "disabled"
            self.update_progress(0)
            self.percentage_label.pack_forget()  # Hide percentage
            self.file_info_frame.pack_forget()   # Hide file info
            self.update_status("Loading point cloud file...", None)
            
            thread = threading.Thread(target=self._process_loading, args=(filename,))
            thread.daemon = True
            thread.start()
    
    def _process_loading(self, filename):
        """
        Loads the point cloud file and updates the UI.
        
        Carga el archivo de nube de puntos y actualiza la interfaz.
        
        Args:
            filename: Path to the file to load.
        """
        try:
            # Get file info first (faster)
            self.update_status("Reading file information...", None)
            self.update_progress(10)
            file_info = io.get_file_info(filename)
            self.update_progress(30)
            
            # Prepare UI with file information
            self.filename_label["text"] = f"Filename: {file_info['filename']} ({file_info['file_size_mb']:.2f} MB)"
            
            # Show point count if available, otherwise show error
            # Mostrar recuento de puntos si está disponible, de lo contrario mostrar error
            if 'point_count' in file_info:
                self.points_label["text"] = f"Points: {file_info['point_count']:,}"
            else:
                self.points_label["text"] = "Points: Unknown"
                if 'header_error' in file_info:
                    self.update_status(f"Warning: {file_info['header_error']}", "warning")
            
            # Load the full point cloud
            self.update_status("Loading point cloud data...", None)
            self.update_progress(50)
            points, summary = io.load_point_cloud(filename)
            self.update_progress(90)
            
            # Store loaded data in controller
            self.controller.point_cloud = points
            self.controller.point_cloud_summary = summary
            
            # Update UI with more detailed information
            self.dimensions_label["text"] = (
                f"Dimensions: X: {summary['x_range'][0]:.2f} to {summary['x_range'][1]:.2f}, "
                f"Y: {summary['y_range'][0]:.2f} to {summary['y_range'][1]:.2f}, "
                f"Z: {summary['z_range'][0]:.2f} to {summary['z_range'][1]:.2f}"
            )
            self.density_label["text"] = f"Density: {summary['point_density']:.2f} points/m²"
            
            # Show success message and enable buttons
            self.update_progress(100)
            # Show percentage label (already positioned correctly)
            self.percentage_label.pack()
            self.file_info_frame.pack(fill=tk.X, padx=10, pady=(5, 10))  # Show file info
            
            # Get just the filename without path for display
            # Obtener solo el nombre del archivo sin la ruta para mostrar
            file_name = os.path.basename(filename)
            self.update_status(f"File '{file_name}' loaded successfully", True)
            
            # Enable buttons
            self.preview_button["state"] = "normal"
            self.next_button["state"] = "normal"
            self.remove_button["state"] = "normal"
            
        except FileNotFoundError:
            self.update_status("Error: File not found", False)
            self.load_button["state"] = "normal"
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", False)
            self.load_button["state"] = "normal"
        except Exception as e:
            self.update_status(f"Error loading file: {str(e)}", False)
            self.load_button["state"] = "normal"
            # Log the error for debugging
            print(f"Error loading point cloud: {str(e)}")
            traceback.print_exc()
    
    def _preview_point_cloud(self):
        """
        Visualizes the point cloud in a separate Open3D window.
        
        Visualiza la nube de puntos en una ventana separada de Open3D.
        """
        if self.controller.point_cloud is None:
            messagebox.showerror("Error", "No point cloud data available")
            return
        
        self.update_status("Opening point cloud preview...", None)
        
        # Import visualization in method to avoid circular imports
        # Importar visualización en el método para evitar importaciones circulares
        from src import visualization
        
        # Run visualization in a separate thread to keep UI responsive
        # Ejecutar visualización en un hilo separado para mantener la interfaz responsiva
        def run_visualization():
            try:
                # Get point cloud from controller
                # Obtener nube de puntos del controlador
                points = self.controller.point_cloud
                
                # Downsample for preview if needed
                # Submuestrear para previsualización si es necesario
                if len(points) > 500000:
                    self.update_status("Downsampling point cloud for preview...", None)
                    points = visualization.downsample_point_cloud(points, target_points=500000)
                
                # Visualize the point cloud
                # Visualizar la nube de puntos
                file_name = os.path.basename(self.controller.input_path.get())
                window_title = f"LiDAR PlotSafe - Preview: {file_name}"
                visualization.visualize_point_cloud(points, title=window_title)
                
                # Update status when visualization window is closed
                # Actualizar estado cuando se cierre la ventana de visualización
                self.update_status(f"Preview closed: {file_name}", None)
                
            except Exception as e:
                error_msg = f"Error in visualization: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                self.update_status(error_msg, False)
        
        # Start visualization thread
        # Iniciar hilo de visualización
        threading.Thread(target=run_visualization).start()
    
    def _next_step(self):
        """
        Proceeds to the next processing stage.
        
        Proceeds to the next processing stage.
        """
        self.controller.show_frame("ParametersFrame")
    
    def _remove_file(self):
        """
        Removes the currently loaded file and resets the interface.
        
        Quita el archivo cargado actualmente y restablece la interfaz.
        
        This allows the user to select a different file without restarting the application.
        Esto permite al usuario seleccionar un archivo diferente sin reiniciar la aplicación.
        """
        # Reset controller data
        # Restablecer los datos del controlador
        self.controller.input_path.set("")
        self.controller.point_cloud = None
        self.controller.point_cloud_summary = None
        
        # Reset UI elements
        # Restablecer elementos de la interfaz
        self.filename_label["text"] = "Filename: "
        self.points_label["text"] = "Points: "
        self.dimensions_label["text"] = "Dimensions: "
        self.density_label["text"] = "Density: "
        
        # Hide information frame
        # Ocultar marco de información
        self.file_info_frame.pack_forget()
        
        # Reset progress bar and status
        # Restablecer barra de progreso y estado
        self.progress_bar["value"] = 0
        if hasattr(self, 'percentage_label') and self.percentage_label.winfo_ismapped():
            self.percentage_label.pack_forget()
        self.update_status("File removed. Ready to load a new file.", None)
        
        # Disable action buttons and re-enable load button
        # Desactivar botones de acción y reactivar botón de carga
        self.preview_button["state"] = "disabled"
        self.next_button["state"] = "disabled"
        self.remove_button["state"] = "disabled"
        self.load_button["state"] = "normal"
    
    def update_progress(self, value):
        """
        Updates the progress bar.
        
        Updates the progress bar.
        
        Args:
            value: Value of the progress bar (0-100).
        """
        self.progress_bar["value"] = value
        self.update_idletasks()
        
        # Only show percentage at 100%
        if value == 100:
            self.percentage_label["text"] = "100%"
        else:
            self.percentage_label.pack_forget()  # Hide percentage if not 100%
    
    def update_status(self, message, success=None):
        """
        Updates the status message.
        
        Updates the status message.
        
        Args:
            message: Message to display.
            success: True if the operation was successful, False if it failed, None for neutral state.
        """
        self.status_label["text"] = message
        self.status_label["anchor"] = "center"  # Center text horizontally
        self.status_label["justify"] = "center"  # Center multiline text
        
        if success is True:
            self.status_label["foreground"] = "green"
        elif success is False:
            self.status_label["foreground"] = "red"
        else:
            self.status_label["foreground"] = self.controller.ACCENT_COLOR
            
        # Ensure the status label is visible
        if not self.status_label.winfo_ismapped():
            self.status_label.pack(fill=tk.X, padx=20, pady=(5, 10))
            
        self.update_idletasks()

    def _create_directory_structure(self, base_dir, structure):
        """
        Recursively creates directory structure from a nested dictionary.
        
        Crea recursivamente la estructura de directorios a partir de un diccionario anidado.
        
        Args:
            base_dir (str): Base directory path to create structure in
            structure (dict): Dictionary with directory names as keys and nested dictionaries as subdirectories
        """
        for dir_name, subdirs in structure.items():
            # Create full path for current directory
            # Crear ruta completa para el directorio actual
            path = os.path.join(base_dir, dir_name)
            
            # Create the directory if it doesn't exist
            # Crear el directorio si no existe
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except Exception as e:
                    raise Exception(f"Failed to create directory {path}: {str(e)}")
            
            # Recursively create subdirectories
            # Crear subdirectorios recursivamente
            if isinstance(subdirs, dict):
                self._create_directory_structure(path, subdirs)

    def _set_project_directory(self):
        """
        Sets the project directory and creates the necessary folder structure.
        
        Establece el directorio del proyecto y crea la estructura de carpetas necesaria.
        """
        # Open directory dialog
        # Abrir diálogo de directorio
        directory = filedialog.askdirectory(
            title="Select project directory"
        )
        
        if directory:
            try:
                # Save the directory path
                # Guardar la ruta del directorio
                self.controller.project_dir.set(directory)
                
                # Define the directory structure
                # Definir la estructura de directorios
                structure = {
                    "inputs": {"raw_point_clouds": {}},
                    "processed": {
                        "cropped": {},
                        "classified": {},
                        "reports": {},
                        "plot_outputs": {},
                        "trees": {}
                    },
                    "temp": {}
                }
                
                # Create the directory structure
                # Crear la estructura de directorios
                self.update_status("Creating project structure...", None)
                self._create_directory_structure(directory, structure)
                
                # Set project structure created flag
                # Establecer la bandera de estructura del proyecto creada
                self.controller.project_structure_created = True
                
                # Enable the load button
                # Habilitar el botón de carga
                self.load_button["state"] = "normal"
                
                # Update status
                # Actualizar estado
                self.update_status(f"Project directory set to: {directory}", True)
                
            except Exception as e:
                # Show error message
                # Mostrar mensaje de error
                messagebox.showerror("Error", f"Error creating project structure: {str(e)}")
                self.update_status(f"Error creating project structure: {str(e)}", False)


class ParametersFrame(ttk.Frame):
    """
    Frame for configuring processing parameters.
    
    Frame for configuring processing parameters.
    """
    
    def __init__(self, parent, controller):
        """
        Initializes the parameters frame.
        
        Initializes the parameters frame.
        
        Args:
            parent: Parent widget.
            controller: Main application controller.
        """
        super().__init__(parent)
        self.controller = controller
        
        # Create widgets
        label = ttk.Label(self, text="Parameters configuration (coming soon)", font=("Arial", 14))
        label.pack(pady=50)
        
        back_button = ttk.Button(
            self, 
            text="← Back",
            command=lambda: controller.show_frame("LoadFrame")
        )
        back_button.pack()


def main():
    """
    Main entry point for the application.
    
    Main entry point for the application.
    """
    app = LiDARPlotSafeLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()