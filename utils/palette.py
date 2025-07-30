def get_color_palette():
    color_map = [
        "#FF5733",  # Red
        "#33FF57",  # Green
        "#3357FF",  # Blue
        "#F1C40F",  # Yellow
        "#8E44AD",  # Purple
        "#E67E22",  # Orange
        "#2ECC71",  # Emerald
        "#3498DB",  # Sky Blue
        "#9B59B6",  # Amethyst
        "#F39C12",  # Carrot
        "#D35400",  # Pumpkin
        "#1ABC9C",  # Turquoise
        "#34495E",  # Wet Asphalt
        "#7F8C8D",  # Concrete
        "#C0392B",  # Alizarin
        "#16A085",  # Green Sea
        "#2980B9",  # Bright Blue
        "#8E44AD",  # Dark Purple
        "#F1C40F",  # Bright Yellow
        "#E74C3C",  # Bright Red
        "#2C3E50",  # Dark Blue
    ]
    
    # Convert hex colors to RGB tuples
    color_map = [tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5)) for color in color_map]
    
    return color_map