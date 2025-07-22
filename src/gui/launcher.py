#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

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
from src.processing import crop_circular_plot

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
        for F in (LoadFrame, ParametersFrame, SegmentationParametersFrame, ClassificationFrame):
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
            self.controller.raw_file_link.set(filename)  # Set file path in controller variable
            
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
        
        Inicializa el marco de parámetros.
        
        Args:
            parent: Parent widget.
            controller: Main application controller.
        """
        super().__init__(parent)
        self.controller = controller
        
        # Status message frame (at the bottom of the window)
        # Marco de mensaje de estado (en la parte inferior de la ventana)
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)
        
        # Status message (initially empty)
        # Mensaje de estado (inicialmente vacío)
        self.status_label = ttk.Label(
            status_frame, 
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
        self.status_label.pack(fill=tk.X)
        self.status_label.pack_forget()  # Hidden initially
        
        # Create content frame with proper styling
        # Crear marco de contenido con estilo adecuado
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title for the parameters section
        # Título para la sección de parámetros
        title_label = ttk.Label(
            self.content_frame,
            text="Plot Cropping Parameters",
            font=("Arial", 14, "bold"),
            foreground=self.controller.TEXT_COLOR,
            background=self.controller.CONTENT_COLOR
        )
        title_label.pack(pady=(5, 15))
        
        # Instructions label
        # Etiqueta de instrucciones
        instructions = ttk.Label(
            self.content_frame,
            text="Set the circular crop parameters for your point cloud.\nA circular area will be extracted based on these coordinates.",
            justify="center",
            background=self.controller.CONTENT_COLOR
        )
        instructions.pack(pady=(0, 15))
        
        # Form frame for parameters input
        # Marco de formulario para entrada de parámetros
        form_frame = ttk.Frame(self.content_frame)
        form_frame.pack(fill=tk.X, padx=50, pady=10)
        
        # Variables for storing the parameter values
        # Variables para almacenar los valores de los parámetros
        self.x_center = tk.DoubleVar(value=0.0)
        self.y_center = tk.DoubleVar(value=0.0)
        self.radius = tk.DoubleVar(value=11.28)  # Default circular plot radius for forestry (approx. 400 m²)
        
        # Create form fields with labels
        # Crear campos de formulario con etiquetas
        
        # X coordinate
        x_frame = ttk.Frame(form_frame)
        x_frame.pack(fill=tk.X, pady=5)
        
        x_label = ttk.Label(
            x_frame,
            text="Center X coordinate (m):",
            width=25,
            anchor="e"
        )
        x_label.pack(side=tk.LEFT, padx=(0, 5))
        
        x_entry = ttk.Entry(
            x_frame,
            textvariable=self.x_center,
            width=15
        )
        x_entry.pack(side=tk.LEFT)
        
        # Y coordinate
        y_frame = ttk.Frame(form_frame)
        y_frame.pack(fill=tk.X, pady=5)
        
        y_label = ttk.Label(
            y_frame,
            text="Center Y coordinate (m):",
            width=25,
            anchor="e"
        )
        y_label.pack(side=tk.LEFT, padx=(0, 5))
        
        y_entry = ttk.Entry(
            y_frame,
            textvariable=self.y_center,
            width=15
        )
        y_entry.pack(side=tk.LEFT)
        
        # Radius
        radius_frame = ttk.Frame(form_frame)
        radius_frame.pack(fill=tk.X, pady=5)
        
        radius_label = ttk.Label(
            radius_frame,
            text="Circle radius (m):",
            width=25,
            anchor="e"
        )
        radius_label.pack(side=tk.LEFT, padx=(0, 5))
        
        radius_entry = ttk.Entry(
            radius_frame,
            textvariable=self.radius,
            width=15
        )
        radius_entry.pack(side=tk.LEFT)
        
        # Information about default values
        # Información sobre valores predeterminados
        info_label = ttk.Label(
            form_frame,
            text="Note: Default radius of 11.28m creates a circular plot of approximately 400 m²",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        info_label.pack(pady=(5, 15))
        
        # Progress bar and percentage in same frame
        # Barra de progreso y porcentaje en el mismo frame
        progress_frame = ttk.Frame(self.content_frame)
        progress_frame.pack(fill=tk.X, padx=50, pady=(5, 15))

        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=450,
            mode="determinate"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Percentage label positioned to the right of progress bar
        # Etiqueta de porcentaje posicionada a la derecha de la barra de progreso
        self.percentage_label = ttk.Label(
            progress_frame,
            text="0%",
            font=("Arial", 10, "bold"),
            foreground="green",
            width=5
        )
        self.percentage_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.percentage_label.pack_forget()  # Initially hidden
        
        # Navigation buttons at the bottom
        # Botones de navegación en la parte inferior
        button_frame = ttk.Frame(self.content_frame)
        button_frame.pack(fill=tk.X, padx=50, pady=20)
        
        # Left side buttons
        # Botones del lado izquierdo
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT, fill=tk.X)
        
        # Back button (left side)
        # Botón volver (lado izquierdo)
        self.back_button = ttk.Button(
            left_buttons, 
            text="← Back",
            command=lambda: controller.show_frame("LoadFrame")
        )
        self.back_button.pack(side=tk.LEFT)
        
        # Right side buttons
        # Botones del lado derecho
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT, fill=tk.X)
        
        # Process button (right side)
        # Botón procesar (lado derecho)
        self.process_button = ttk.Button(
            right_buttons, 
            text="Crop Point Cloud",
            command=self._process_crop
        )
        self.process_button.pack(side=tk.LEFT)
        
        # Next button (initially disabled)
        # Botón siguiente (inicialmente deshabilitado)
        self.next_button = ttk.Button(
            right_buttons, 
            text="Next →",
            command=lambda: controller.show_frame("SegmentationParametersFrame"),
            state="disabled"
        )
        self.next_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Results frame (initially hidden)
        # Marco de resultados (inicialmente oculto)
        self.results_frame = ttk.LabelFrame(
            self.content_frame,
            text="Processing Results",
            padding=10
        )
        
        # Define result label references for later updating
        # Definir referencias de etiquetas de resultados para actualización posterior
        self.result_labels = {}
    
    def _process_crop(self):
        """
        Process the point cloud cropping based on parameters.
        
        Procesa el recorte de la nube de puntos según los parámetros.
        """
        # Check if we have a raw file to process
        # Verificar si tenemos un archivo sin procesar para procesar
        if not self.controller.raw_file_link.get():
            self.update_status("No point cloud file loaded. Go back and load a file first.", False)
            return
        
        # Get parameters from form
        # Obtener parámetros del formulario
        center_x = self.x_center.get()
        center_y = self.y_center.get()
        radius = self.radius.get()
        
        # Validate parameters
        # Validar parámetros
        if radius <= 0:
            self.update_status("Radius must be positive", False)
            return
        
        # Set input and output paths
        # Establecer rutas de entrada y salida
        input_file = self.controller.raw_file_link.get()
        
        # If the link is a reference file, get the actual path
        # Si el enlace es un archivo de referencia, obtener la ruta real
        if input_file.endswith('.reference'):
            with open(input_file, 'r') as f:
                input_file = f.read().strip()
        
        # Set output path in the project structure
        # Establecer ruta de salida en la estructura del proyecto
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(
            self.controller.project_dir.get(),
            "processed",
            "cropped",
            f"{name_without_ext}_cropped.las"
        )
        
        # Update status and show progress animation
        # Actualizar estado y mostrar animación de progreso
        self.update_status("Processing point cloud crop...", None)
        
        # Start processing in a separate thread to keep UI responsive
        # Iniciar procesamiento en un hilo separado para mantener la interfaz responsiva
        thread = threading.Thread(
            target=self._run_processing,
            args=(input_file, output_file, center_x, center_y, radius)
        )
        thread.daemon = True
        thread.start()
    
    def _run_processing(self, input_file, output_file, center_x, center_y, radius):
        """
        Run the processing operation in a separate thread.
        
        Ejecutar la operación de procesamiento en un hilo separado.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius: Radius of circle
        """
        try:
            # Update progress intermittently
            # Actualizar progreso intermitentemente
            for i in range(0, 101, 10):
                if i == 0:
                    # Starting...
                    # Iniciando...
                    self.controller.update_progress(i)
                elif i < 30:
                    # Reading file
                    # Leyendo archivo
                    self.controller.update_progress(i, "Reading point cloud...")
                    time.sleep(0.2)  # Simulate processing time
                elif i < 70:
                    # Processing
                    # Procesando
                    self.controller.update_progress(i, "Applying circular crop...")
                    time.sleep(0.3)  # Simulate processing time
                elif i < 90:
                    # Saving
                    # Guardando
                    self.controller.update_progress(i, "Writing cropped point cloud...")
                    time.sleep(0.2)  # Simulate processing time
                else:
                    # Finishing
                    # Finalizando
                    self.controller.update_progress(i, "Finalizing...")
                    time.sleep(0.1)  # Simulate processing time
            
            # Perform the actual cropping operation
            # Realizar la operación de recorte real
            stats = crop_circular_plot(
                input_file,
                output_file,
                center_x,
                center_y,
                radius
            )
            
            # Reset progress
            # Reiniciar progreso
            self.controller.update_progress(0)
            
            # Show success message with results
            # Mostrar mensaje de éxito con resultados
            self._show_results(stats, output_file)
            
        except Exception as e:
            # Show error
            # Mostrar error
            self.controller.update_progress(0)
            self.update_status(f"Error during crop: {str(e)}", False)
    
    def _show_results(self, stats, output_file):
        """
        Display processing results.
        
        Muestra resultados del procesamiento.
        
        Args:
            stats: Statistics from processing
            output_file: Path to output file
        """
        # Clear any previous results
        # Limpiar resultados previos
        if hasattr(self, 'results_frame') and self.results_frame.winfo_ismapped():
            self.results_frame.pack_forget()
            for widget in self.results_frame.winfo_children():
                widget.destroy()
        
        # Setup results frame
        # Configurar marco de resultados
        self.results_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        
        # Add result information
        # Añadir información de resultados
        result_items = [
            ("Total points", f"{stats['total_points']:,}"),
            ("Points kept", f"{stats['percent_kept']:.1f}%"),
            ("Area", f"{stats['area_m2']:.1f} m²"),
            ("Density", f"{stats['density_pts_m2']:.1f} points/m²"),
            ("Output file", os.path.basename(output_file)),
        ]
        
        # Create grid of results
        # Crear cuadrícula de resultados
        for i, (label, value) in enumerate(result_items):
            name_label = ttk.Label(self.results_frame, text=label, font=("Arial", 10))
            name_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            value_label = ttk.Label(self.results_frame, text=value, font=("Arial", 10, "bold"))
            value_label.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            
            # Store reference
            # Almacenar referencia
            self.result_labels[label] = value_label
        
        # Add preview button for the cropped point cloud
        # Añadir botón de previsualización para la nube de puntos recortada
        preview_button = ttk.Button(
            self.results_frame,
            text="Preview cropped result",
            command=lambda: self._preview_cropped_point_cloud(output_file)
        )
        preview_button.grid(row=len(result_items), column=0, columnspan=2, pady=(10, 5))
        
        # Update status and enable the next button
        # Actualizar estado y habilitar el botón siguiente
        self.update_status("Point cloud successfully cropped! You can proceed to the next step.", True)
        
        # Store reference to cropped file path in controller for next steps
        # Almacenar referencia a la ruta del archivo recortado en el controlador para pasos siguientes
        self.controller.cropped_file_path = output_file
        
        self.next_button.config(state="normal")
    
    def _preview_cropped_point_cloud(self, output_file):
        """
        Visualizes the cropped point cloud in a separate Open3D window.
        
        Visualiza la nube de puntos recortada en una ventana separada de Open3D.
        """
        self.update_status("Opening cropped point cloud preview...", None)
        
        # Import visualization in method to avoid circular imports
        # Importar visualización en el método para evitar importaciones circulares
        from src import visualization
        
        # Run visualization in a separate thread to keep UI responsive
        # Ejecutar visualización en un hilo separado para mantener la interfaz responsiva
        def run_visualization():
            try:
                # Load the cropped point cloud
                # Cargar la nube de puntos recortada
                points, _ = io.load_point_cloud(output_file)
                
                # Downsample for preview if needed
                # Submuestrear para previsualización si es necesario
                if len(points) > 500000:
                    self.update_status("Downsampling cropped point cloud for preview...", None)
                    points = visualization.downsample_point_cloud(points, target_points=500000)
                
                # Visualize the point cloud
                # Visualizar la nube de puntos
                file_name = os.path.basename(output_file)
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
    
    def update_status(self, message, success=None):
        """
        Updates the status message.
        
        Actualiza el mensaje de estado.
        
        Args:
            message: Message to display.
            success: True if the operation was successful, False if it failed, None for neutral state.
        """
        # Show status message and set color based on success
        # Mostrar mensaje de estado y establecer color según el éxito
        self.status_label["text"] = message
        
        if success is True:
            self.status_label["foreground"] = "green"
        elif success is False:
            self.status_label["foreground"] = "red"
        else:
            self.status_label["foreground"] = self.controller.ACCENT_COLOR
            
        # Ensure the status label is visible
        # Asegurar que la etiqueta de estado sea visible
        if not self.status_label.winfo_ismapped():
            self.status_label.pack(fill=tk.X)
    
    def update_progress(self, value):
        """
        Updates the progress bar.
        
        Actualiza la barra de progreso.
        
        Args:
            value: Value of the progress bar (0-100).
        """
        # Update progress bar value
        # Actualizar valor de la barra de progreso
        self.progress_bar["value"] = value
        
        # Show/hide percentage label based on progress
        # Mostrar/ocultar etiqueta de porcentaje según el progreso
        if value > 0:
            self.percentage_label["text"] = f"{int(value)}%"
            if not self.percentage_label.winfo_ismapped():
                self.percentage_label.pack(side=tk.RIGHT, padx=(5, 0))
        else:
            if self.percentage_label.winfo_ismapped():
                self.percentage_label.pack_forget()

class SegmentationParametersFrame(ttk.Frame):
    """
    Frame for configuring tree segmentation parameters.
    
    Frame para configurar los parámetros de segmentación de árboles.
    """
    
    def __init__(self, parent, controller):
        """
        Initializes the tree segmentation parameters frame.
        
        Inicializa el marco de parámetros de segmentación de árboles.
        
        Args:
            parent: Parent widget.
            controller: Main application controller.
        """
        super().__init__(parent)
        self.controller = controller
        
        # Status message frame (at the bottom of the window)
        # Marco de mensaje de estado (en la parte inferior de la ventana)
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)
        
        # Status message (initially empty)
        # Mensaje de estado (inicialmente vacío)
        self.status_label = ttk.Label(
            status_frame, 
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
        self.status_label.pack(fill=tk.X)
        self.status_label.pack_forget()  # Hidden initially
        
        # Create content frame with proper styling
        # Crear marco de contenido con estilo adecuado
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title for the segmentation parameters section
        # Título para la sección de parámetros de segmentación
        title_label = ttk.Label(
            self.content_frame,
            text="Tree Segmentation Parameters",
            font=("Arial", 14, "bold"),
            foreground=self.controller.TEXT_COLOR,
            background=self.controller.CONTENT_COLOR
        )
        title_label.pack(pady=(5, 15))
        
        # Instructions label
        # Etiqueta de instrucciones
        instructions = ttk.Label(
            self.content_frame,
            text="Configure parameters for tree segmentation in your point cloud.\nThese settings affect how individual trees are detected in the plot.",
            justify="center",
            background=self.controller.CONTENT_COLOR
        )
        instructions.pack(pady=(0, 15))
        
        # Create a notebook for organizing parameter groups
        # Crear un notebook para organizar grupos de parámetros
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Create frames for different parameter categories
        # Crear marcos para diferentes categorías de parámetros
        self.downsampling_frame = ttk.Frame(self.notebook)
        self.clustering_frame = ttk.Frame(self.notebook)
        self.filtering_frame = ttk.Frame(self.notebook)
        
        # Add frames to notebook with tabs
        # Añadir marcos al notebook con pestañas
        self.notebook.add(self.downsampling_frame, text="Downsampling")
        self.notebook.add(self.clustering_frame, text="Clustering")
        self.notebook.add(self.filtering_frame, text="Tree Filtering")
        
        # Configure parameter variables
        # Configurar variables de parámetros
        self.voxel_size = tk.DoubleVar(value=0.05)  # Default voxel size in meters
        self.eps = tk.DoubleVar(value=0.2)  # Default eps for DBSCAN in meters
        self.min_samples = tk.IntVar(value=5)  # Default min_samples for DBSCAN
        self.slice_height = tk.DoubleVar(value=1.3)  # Default height of horizontal slice (DBH height) in meters
        self.slice_thickness = tk.DoubleVar(value=0.2)  # Default thickness of horizontal slice in meters
        self.min_tree_height = tk.DoubleVar(value=1.5)  # Default minimum tree height in meters
        self.min_points = tk.IntVar(value=50)  # Default minimum points per tree
        self.auto_normalize = tk.BooleanVar(value=True)  # Default auto-normalization enabled
        
        # Create form fields for downsampling parameters
        # Crear campos de formulario para parámetros de submuestreo
        self._create_downsampling_form()
        
        # Create form fields for clustering parameters
        # Crear campos de formulario para parámetros de clustering
        self._create_clustering_form()
        
        # Create form fields for tree filtering parameters
        # Crear campos de formulario para parámetros de filtrado de árboles
        self._create_filtering_form()
        
        # Progress bar and percentage in same frame
        # Barra de progreso y porcentaje en el mismo frame
        progress_frame = ttk.Frame(self.content_frame)
        progress_frame.pack(fill=tk.X, padx=50, pady=(15, 15))

        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=450,
            mode="determinate"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Percentage label positioned to the right of progress bar
        # Etiqueta de porcentaje posicionada a la derecha de la barra de progreso
        self.percentage_label = ttk.Label(
            progress_frame,
            text="0%",
            font=("Arial", 10, "bold"),
            foreground="green",
            width=5
        )
        self.percentage_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.percentage_label.pack_forget()  # Initially hidden
        
        # Navigation buttons at the bottom
        # Botones de navegación en la parte inferior
        button_frame = ttk.Frame(self.content_frame)
        button_frame.pack(fill=tk.X, padx=50, pady=20)
        
        # Left side buttons
        # Botones del lado izquierdo
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT, fill=tk.X)
        
        # Back button (left side)
        # Botón volver (lado izquierdo)
        self.back_button = ttk.Button(
            left_buttons, 
            text="← Back",
            command=lambda: controller.show_frame("ParametersFrame")
        )
        self.back_button.pack(side=tk.LEFT)
        
        # Right side buttons
        # Botones del lado derecho
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT, fill=tk.X)
        
        # Process button (right side)
        # Botón procesar (lado derecho)
        self.process_button = ttk.Button(
            right_buttons, 
            text="Run Tree Segmentation",
            command=self._process_segmentation
        )
        self.process_button.pack(side=tk.LEFT)
        
        # Next button (initially disabled)
        # Botón siguiente (inicialmente deshabilitado)
        self.next_button = ttk.Button(
            right_buttons, 
            text="Next →",
            command=lambda: controller.show_frame("ClassificationFrame"),
            state="disabled"
        )
        self.next_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Results frame (initially hidden)
        # Marco de resultados (inicialmente oculto)
        self.results_frame = ttk.LabelFrame(
            self.content_frame,
            text="Segmentation Results",
            padding=10
        )
        
        # Define result label references for later updating
        # Definir referencias de etiquetas de resultados para actualización posterior
        self.result_labels = {}

    def _create_downsampling_form(self):
        """
        Creates form elements for downsampling parameters.
        
        Crea elementos de formulario para parámetros de submuestreo.
        """
        # Voxel size parameter
        # Parámetro de tamaño de voxel
        voxel_frame = ttk.Frame(self.downsampling_frame)
        voxel_frame.pack(fill=tk.X, pady=15, padx=20)
        
        voxel_label = ttk.Label(
            voxel_frame,
            text="Voxel size (m):",
            width=25,
            anchor="e"
        )
        voxel_label.pack(side=tk.LEFT, padx=(0, 5))
        
        voxel_entry = ttk.Entry(
            voxel_frame,
            textvariable=self.voxel_size,
            width=15
        )
        voxel_entry.pack(side=tk.LEFT)
        
        voxel_info = ttk.Label(
            self.downsampling_frame,
            text="Smaller values preserve more detail but increase processing time.\nRecommended range: 0.01 - 0.1 meters.",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        voxel_info.pack(padx=20, pady=(0, 10))
    
    def _create_clustering_form(self):
        """
        Creates form elements for clustering parameters.
        
        Crea elementos de formulario para parámetros de clustering.
        """
        # DBSCAN eps parameter
        # Parámetro eps de DBSCAN
        eps_frame = ttk.Frame(self.clustering_frame)
        eps_frame.pack(fill=tk.X, pady=(15, 5), padx=20)
        
        eps_label = ttk.Label(
            eps_frame,
            text="DBSCAN distance (eps, m):",
            width=25,
            anchor="e"
        )
        eps_label.pack(side=tk.LEFT, padx=(0, 5))
        
        eps_entry = ttk.Entry(
            eps_frame,
            textvariable=self.eps,
            width=15
        )
        eps_entry.pack(side=tk.LEFT)
        
        # DBSCAN min_samples parameter
        # Parámetro min_samples de DBSCAN
        min_samples_frame = ttk.Frame(self.clustering_frame)
        min_samples_frame.pack(fill=tk.X, pady=5, padx=20)
        
        min_samples_label = ttk.Label(
            min_samples_frame,
            text="Min. points per cluster:",
            width=25,
            anchor="e"
        )
        min_samples_label.pack(side=tk.LEFT, padx=(0, 5))
        
        min_samples_entry = ttk.Entry(
            min_samples_frame,
            textvariable=self.min_samples,
            width=15
        )
        min_samples_entry.pack(side=tk.LEFT)
        
        # Slice height parameter
        # Parámetro de altura de rebanada
        slice_height_frame = ttk.Frame(self.clustering_frame)
        slice_height_frame.pack(fill=tk.X, pady=5, padx=20)
        
        slice_height_label = ttk.Label(
            slice_height_frame,
            text="Horizontal slice height (m):",
            width=25,
            anchor="e"
        )
        slice_height_label.pack(side=tk.LEFT, padx=(0, 5))
        
        slice_height_entry = ttk.Entry(
            slice_height_frame,
            textvariable=self.slice_height,
            width=15
        )
        slice_height_entry.pack(side=tk.LEFT)
        
        # Slice thickness parameter
        # Parámetro de grosor de rebanada
        slice_thickness_frame = ttk.Frame(self.clustering_frame)
        slice_thickness_frame.pack(fill=tk.X, pady=5, padx=20)
        
        slice_thickness_label = ttk.Label(
            slice_thickness_frame,
            text="Slice thickness (m):",
            width=25,
            anchor="e"
        )
        slice_thickness_label.pack(side=tk.LEFT, padx=(0, 5))
        
        slice_thickness_entry = ttk.Entry(
            slice_thickness_frame,
            textvariable=self.slice_thickness,
            width=15
        )
        slice_thickness_entry.pack(side=tk.LEFT)
        
        # Information about clustering parameters
        # Información sobre parámetros de clustering
        clustering_info = ttk.Label(
            self.clustering_frame,
            text="The horizontal slice is taken at the specified height (typically DBH height, 1.3m).\nClustering is performed on this slice to identify tree trunks.",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        clustering_info.pack(padx=20, pady=(5, 10))
    
    def _create_filtering_form(self):
        """
        Creates form elements for tree filtering parameters.
        
        Crea elementos de formulario para parámetros de filtrado de árboles.
        """
        # Minimum tree height parameter
        # Parámetro de altura mínima de árbol
        min_height_frame = ttk.Frame(self.filtering_frame)
        min_height_frame.pack(fill=tk.X, pady=(15, 5), padx=20)
        
        min_height_label = ttk.Label(
            min_height_frame,
            text="Min. tree height (m):",
            width=25,
            anchor="e"
        )
        min_height_label.pack(side=tk.LEFT, padx=(0, 5))
        
        min_height_entry = ttk.Entry(
            min_height_frame,
            textvariable=self.min_tree_height,
            width=15
        )
        min_height_entry.pack(side=tk.LEFT)
        
        # Minimum points per tree parameter
        # Parámetro de puntos mínimos por árbol
        min_points_frame = ttk.Frame(self.filtering_frame)
        min_points_frame.pack(fill=tk.X, pady=5, padx=20)
        
        min_points_label = ttk.Label(
            min_points_frame,
            text="Min. points per tree:",
            width=25,
            anchor="e"
        )
        min_points_label.pack(side=tk.LEFT, padx=(0, 5))
        
        min_points_entry = ttk.Entry(
            min_points_frame,
            textvariable=self.min_points,
            width=15
        )
        min_points_entry.pack(side=tk.LEFT)
        
        # Information about filtering parameters
        # Información sobre parámetros de filtrado
        filtering_info = ttk.Label(
            self.filtering_frame,
            text="Tree height is calculated using the full point cloud around each cluster.\nClusters that don't meet these minimums will be filtered out.",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        filtering_info.pack(padx=20, pady=(5, 10))
    
    def _process_segmentation(self):
        """
        Process tree segmentation based on parameters.
        
        Procesa la segmentación de árboles según los parámetros.
        """
        # Check if we have a cropped file to process
        # Verificar si tenemos un archivo recortado para procesar
        if not hasattr(self.controller, 'cropped_file_path') or not self.controller.cropped_file_path:
            self.update_status("No cropped point cloud available. Complete the previous step first.", False)
            return
        
        # Get parameters from form
        # Obtener parámetros del formulario
        voxel_size = self.voxel_size.get()
        eps = self.eps.get()
        min_samples = self.min_samples.get()
        slice_height = self.slice_height.get()
        slice_thickness = self.slice_thickness.get()
        min_tree_height = self.min_tree_height.get()
        min_points = self.min_points.get()
        auto_normalize = self.auto_normalize.get()
        
        # Validate parameters
        # Validar parámetros
        if voxel_size <= 0:
            self.update_status("Voxel size must be positive", False)
            return
        if eps <= 0:
            self.update_status("DBSCAN distance must be positive", False)
            return
        if min_samples <= 0:
            self.update_status("Min points per cluster must be positive", False)
            return
        if slice_thickness <= 0:
            self.update_status("Slice thickness must be positive", False)
            return
        if min_tree_height <= 0:
            self.update_status("Min tree height must be positive", False)
            return
        if min_points <= 0:
            self.update_status("Min points per tree must be positive", False)
            return
        
        # Set input and output paths
        # Establecer rutas de entrada y salida
        input_file = self.controller.cropped_file_path
        
        # Set output path in the project structure
        # Establecer ruta de salida en la estructura del proyecto
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(
            self.controller.project_dir.get(),
            "processed",
            "trees",
            f"{name_without_ext}_trees.las"
        )
        
        # Update status and show progress animation
        # Actualizar estado y mostrar animación de progreso
        self.update_status("Processing tree segmentation...", None)
        
        # Start processing in a separate thread to keep UI responsive
        # Iniciar procesamiento en un hilo separado para mantener la interfaz responsiva
        thread = threading.Thread(
            target=self._run_processing,
            args=(input_file, output_file, voxel_size, eps, min_samples, 
                  slice_height, slice_thickness, min_tree_height, min_points, auto_normalize)
        )
        thread.daemon = True
        thread.start()
    
    def _run_processing(self, input_file, output_file, voxel_size, eps, min_samples, 
                       slice_height, slice_thickness, min_tree_height, min_points, auto_normalize):
        """
        Run the segmentation processing operation in a separate thread.
        
        Ejecutar la operación de procesamiento de segmentación en un hilo separado.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            voxel_size: Voxel size for downsampling
            eps: DBSCAN distance parameter
            min_samples: Minimum points per DBSCAN cluster
            slice_height: Height of horizontal slice
            slice_thickness: Thickness of horizontal slice
            min_tree_height: Minimum tree height to keep
            min_points: Minimum points per tree to keep
            auto_normalize: Auto-normalization flag
        """
        try:
            # Import necessary modules
            # Importar módulos necesarios
            from src.pipeline.segmentation import segment_trees
            from src.io import load_point_cloud
            
            # Load the point cloud from input file
            # Cargar la nube de puntos desde el archivo de entrada
            points, _ = load_point_cloud(input_file)
            
            # Update progress intermittently
            # Actualizar progreso intermitentemente
            for i in range(0, 101, 10):
                if i == 0:
                    # Starting...
                    # Iniciando...
                    self.controller.update_progress(i)
                elif i < 20:
                    # Reading file
                    # Leyendo archivo
                    self.controller.update_progress(i, "Reading point cloud...")
                    time.sleep(0.2)  # Simulate processing time
                elif i < 40:
                    # Downsampling
                    # Submuestreando
                    self.controller.update_progress(i, "Downsampling point cloud...")
                    time.sleep(0.3)  # Simulate processing time
                elif i < 60:
                    # Clustering
                    # Agrupando
                    self.controller.update_progress(i, "Clustering tree trunks...")
                    time.sleep(0.3)  # Simulate processing time
                elif i < 80:
                    # Filtering
                    # Filtrando
                    self.controller.update_progress(i, "Filtering tree clusters...")
                    time.sleep(0.2)  # Simulate processing time
                elif i < 90:
                    # Saving
                    # Guardando
                    self.controller.update_progress(i, "Writing segmented trees...")
                    time.sleep(0.2)  # Simulate processing time
                else:
                    # Finishing
                    # Finalizando
                    self.controller.update_progress(i, "Finalizing...")
                    time.sleep(0.1)  # Simulate processing time
            
            # Perform the actual segmentation operation
            # Realizar la operación de segmentación real
            trees, metadata = segment_trees(
                points,
                voxel_size=voxel_size,
                eps=eps,
                min_samples=min_samples,
                slice_height=slice_height,
                slice_thickness=slice_thickness,
                min_tree_height=min_tree_height,
                min_points=min_points,
                auto_normalize=auto_normalize
            )
            
            # Save the result to the output file
            # Guardar el resultado en el archivo de salida
            from src.io import save_segmented_point_cloud
            save_segmented_point_cloud(trees, output_file)
            
            # Save the result in controller for later use
            # Guardar el resultado en el controlador para uso posterior
            self.controller.segmentation_result = metadata
            self.controller.segmented_file_path = output_file
            
            # Reset progress
            # Reiniciar progreso
            self.controller.update_progress(0)
            
            # Show success message with results
            # Mostrar mensaje de éxito con resultados
            self._show_results(metadata, output_file)
            
        except Exception as e:
            # Show error
            # Mostrar error
            self.controller.update_progress(0)
            self.update_status(f"Error during segmentation: {str(e)}", False)
            traceback.print_exc()
    
    def _show_results(self, result, output_file):
        """
        Display segmentation results.
        
        Muestra resultados de la segmentación.
        
        Args:
            result: Results dictionary from segmentation
            output_file: Path to output file
        """
        # Check if results frame exists
        # Verificar si existe el marco de resultados
        if not hasattr(self, 'results_frame') or not self.results_frame:
            return
            
        # Show results frame
        # Mostrar marco de resultados
        self.results_frame.pack(fill=tk.X, padx=20, pady=10)
            
        # Clear previous results
        # Limpiar resultados anteriores
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Create result labels
        # Crear etiquetas de resultados
        num_trees = result.get("num_trees", 0)
        total_points = result.get("total_points", 0)
        
        # Add file info
        # Añadir información del archivo
        file_label = ttk.Label(
            self.results_frame,
            text=f"Segmented trees saved to: {os.path.basename(output_file)}",
            justify="left",
            anchor="w"
        )
        file_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        # Add trees info
        # Añadir información de árboles
        trees_label = ttk.Label(
            self.results_frame,
            text=f"Trees detected: {num_trees}",
            justify="left",
            anchor="w"
        )
        trees_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        # Add points info
        # Añadir información de puntos
        points_label = ttk.Label(
            self.results_frame,
            text=f"Total points in trees: {total_points:,}",
            justify="left",
            anchor="w"
        )
        points_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        # Enable next button
        # Habilitar botón siguiente
        self.next_button["state"] = "normal"
        
        # Show success message
        # Mostrar mensaje de éxito
        self.update_status(f"Tree segmentation completed successfully. {num_trees} trees detected.", True)
    
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
            self.percentage_label.pack(side=tk.RIGHT, padx=(5, 0))
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
        
class ClassificationFrame(ttk.Frame):
    """
    Frame for point cloud classification and analysis.
    
    Marco para clasificación y análisis de nube de puntos.
    """
    
    def __init__(self, parent, controller):
        """
        Initializes the classification frame.
        
        Inicializa el marco de clasificación.
        
        Args:
            parent: Parent widget.
            controller: Main application controller.
        """
        super().__init__(parent)
        self.controller = controller
        
        # Create content frame with proper styling
        # Crear marco de contenido con estilo adecuado
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title for the classification section
        # Título para la sección de clasificación
        title_label = ttk.Label(
            self.content_frame,
            text="LiDAR Classification Module",
            font=("Arial", 14, "bold"),
            foreground=self.controller.TEXT_COLOR,
            background=self.controller.CONTENT_COLOR
        )
        title_label.pack(pady=(5, 15))
        
        # Australian English message about future module
        # Mensaje en inglés australiano sobre el módulo futuro
        message_frame = ttk.Frame(self.content_frame, style="TFrame")
        message_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)
        
        message_label = ttk.Label(
            message_frame,
            text=(
                "G'day! This section will soon feature our advanced LiDAR classification module.\n\n"
                "Once implemented, you'll be able to segment your plot into individual trees, "
                "calculate trunk diameters and heights, and estimate timber volume with ripper accuracy.\n\n"
                "We're currently working flat out to bring you these bonza features in our next update.\n\n"
                "Check back soon, mate!"
            ),
            font=("Arial", 11),
            justify="center",
            foreground=self.controller.TEXT_COLOR,
            background=self.controller.CONTENT_COLOR,
            wraplength=500
        )
        message_label.pack(pady=20)
        
        # Navigation buttons at the bottom
        # Botones de navegación en la parte inferior
        button_frame = ttk.Frame(self.content_frame)
        button_frame.pack(fill=tk.X, padx=50, pady=20)
        
        # Back button
        # Botón volver
        self.back_button = ttk.Button(
            button_frame, 
            text="← Back",
            command=lambda: controller.show_frame("ParametersFrame")
        )
        self.back_button.pack(side=tk.LEFT)
        
        # Status message (initially empty)
        # Mensaje de estado (inicialmente vacío)
        self.status_label = ttk.Label(
            self.content_frame, 
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
        self.status_label.pack(fill=tk.X, padx=20, pady=(15, 5))
        self.status_label.pack_forget()  # Hidden initially
    
    def update_status(self, message, success=None):
        """
        Updates the status message.
        
        Actualiza el mensaje de estado.
        
        Args:
            message: Message to display.
            success: True if the operation was successful, False if it failed, None for neutral state.
        """
        if not message:
            self.status_label.pack_forget()
            return
            
        self.status_label.pack(fill=tk.X, padx=20, pady=(15, 5))
        self.status_label.config(text=message)
        
        if success is True:
            self.status_label.config(foreground="green")
        elif success is False:
            self.status_label.config(foreground="red")
        else:
            self.status_label.config(foreground=self.controller.ACCENT_COLOR)


def main():
    """
    Main entry point for the application.
    
    Main entry point for the application.
    """
    app = LiDARPlotSafeLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()