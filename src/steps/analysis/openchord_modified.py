import numpy as np
import drawsvg as dw

__version__ = "0.1.7"

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, -y

def norm(x, y):
    return np.sqrt(x*x+y*y)

def get_arc(radius, start_angle, end_angle):
    # calculate coords
    x1, y1 = pol2cart(radius, start_angle)
    x2, y2 = pol2cart(radius, end_angle)
    return x1, y1, x2, y2

def arc(radius, start_angle, end_angle, color="black", opacity=0.9, thickness=0.07):
    # coordinates
    k = 1.0 + thickness
    x1, y1, x2, y2 = get_arc(radius, start_angle, end_angle)
    x3, y3, x4, y4 = k*x2, k*y2, k*x1, k*y1
    # arc direction
    large_arc = 1 if end_angle-start_angle > np.pi else 0
    # create path 
    p = dw.Path(fill=color, fill_opacity=opacity)
    p.M(x1, y1)
    p.A(radius, radius, rot=0, large_arc=large_arc, sweep=0, ex=x2, ey=y2)
    p.L(x3, y3)
    p.A(k*radius, k*radius, rot=0, large_arc=large_arc, sweep=1, ex=x4, ey=y4).Z()
    return p

def thin_ring(radius, start_angle, end_angle, color="black", opacity=0.6, ring_width=3):
    """
    Draw a thin ring for self-dependencies instead of a curved ribbon.
    The ring appears as a thin arc just inside the main arc.
    """
    # Use a slightly smaller radius for the inner ring
    inner_radius = radius - ring_width
    
    # Create a thin arc path
    x1, y1, x2, y2 = get_arc(inner_radius, start_angle, end_angle)
    x3, y3, x4, y4 = get_arc(radius, start_angle, end_angle)
    
    # Arc direction
    large_arc = 1 if end_angle - start_angle > np.pi else 0
    
    # Create path for the ring
    p = dw.Path(fill=color, fill_opacity=opacity)
    p.M(x1, y1)
    p.A(inner_radius, inner_radius, rot=0, large_arc=large_arc, sweep=0, ex=x2, ey=y2)
    p.L(x4, y4)
    p.A(radius, radius, rot=0, large_arc=large_arc, sweep=1, ex=x3, ey=y3)
    p.Z()
    return p

def ribbon(radius, source_a1, source_a2, target_a1, target_a2, color="black", opacity=0.6, control_strength=0.6):
    x1, y1, x2, y2 = get_arc(radius, source_a1, source_a2)
    x3, y3, x4, y4 = get_arc(radius, target_a1, target_a2)
    k = 1.0 - control_strength
    xctr1, yctr1, xctr2, yctr2, xctr3, yctr3, xctr4, yctr4 = k*x1, k*y1, k*x2, k*y2, k*x3, k*y3, k*x4, k*y4
    # arc direction
    source_large_arc = 1 if source_a2 - source_a1 > np.pi else 0
    target_large_arc = 1 if target_a2 - target_a1 > np.pi else 0
    # define path
    p = dw.Path(fill=color, fill_opacity=opacity)
    p.M(x1, y1)
    p.A(radius, radius, rot=0, large_arc=source_large_arc, sweep=0, ex=x2, ey=y2)
    p.C(xctr2, yctr2, xctr3, yctr3, x3, y3)
    p.A(radius, radius, rot=0, large_arc=target_large_arc, sweep=0, ex=x4, ey=y4)
    p.C(xctr4, yctr4, xctr1, yctr1, x1, y1)
    p.Z()
    return p

def is_symmetric(A):
        A = np.array(A)
        m, n = A.shape
        if m != n:
            return False
        return np.allclose(A, A.T)

class Chord:

    colormap_default = ["#FFB6C1", "#FFD700", "#FFA07A", "#90EE90", "#87CEFA", "#DA70D6", "#FF69B4", "#20B2AA"]
    colormap_vibrant = ["#FF6B6B", "#F9844A", "#F9C74F", "#90BE6D", "#43AA8B", "#4D908E", "#577590", "#277DA1"]
    
    # Continent-based color mapping
    continent_colors = {
        "North America": "#1f77b4",      # Blue
        "Central America": "#ff7f0e",    # Orange  
        "South America": "#2ca02c",      # Green
        "Europe": "#d62728",             # Red
        "Asia": "#9467bd",               # Purple
        "Africa": "#8c564b",             # Brown
        "Oceania": "#e377c2"             # Pink
    }
    
    country_to_continent = {
        # North America
        "Canada": "North America",
        "United States": "North America",
        "Mexico": "North America",
        
        # Central America
        "Costa Rica": "Central America",
        "Dominican Republic": "Central America", 
        "Guatemala": "Central America",
        
        # South America
        "Argentina": "South America",
        "Brazil": "South America",
        "Colombia": "South America",
        
        # Europe
        "Germany": "Europe",
        "Spain": "Europe",
        "France": "Europe",
        "United Kingdom": "Europe",
        "Italy": "Europe",
        
        # Asia
        "Indonesia": "Asia",
        "India": "Asia",
        "Japan": "Asia",
        
        # Africa
        "Egypt": "Africa",
        "Nigeria": "Africa",
        "South Africa": "Africa",
        
        # Oceania
        "Australia": "Oceania",
        "New Zealand": "Oceania",
        "Papua New Guinea": "Oceania"
    }
    
    def __init__(self, data, labels=[], scale=[], radius=200):
        self.matrix = np.array(data)
        self.labels = labels
        self.scale = scale if len(scale) == len(labels) else np.ones(len(labels))
        self._radius = radius
        self._padding = 50
        self.plot_area = self.get_plot_area()
        self.font_size = 10
        self.font_family = "Arial"
        self._gap_size = 0.01
        self.ribbon_gap = 0.01
        self.ribbon_stiffness = 0.6
        self._rotation = 0
        self.arc_thickness = 0.07
        self.text_position = 0.1
        self.bg_color = "#ffffff"
        self.bg_transparancy = 1.0
        self.use_continent_colors = True  # Default to False
        self.ring_width = 5  # Width of thin rings for self-dependencies
        self.shape = self.matrix.shape[0]
        self.pairs = self.get_pairs()
        self.row_sum = np.sum(self.matrix, axis=1)
        self.total = np.sum(self.matrix)
        self.conversion_rate = (2*np.pi-self.gap_size*self.shape)/self.total
        self.is_symmetric = is_symmetric(self.matrix)
        self.ideogram_ends = self.get_ideogram_ends()
        self.ribbon_ends = self.get_ribbon_ends()
        self._colormap = self.colormap_default
        self._gradient_style = "default"
        self.gradients = self.get_gradients()

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        self._colormap = value
        self.gradients = self.get_gradients()

    @property
    def gap_size(self):
        return self._gap_size

    @gap_size.setter
    def gap_size(self, value):
        self._gap_size = value
        self.conversion_rate = (2*np.pi-self.gap_size*self.shape)/self.total
        self.ideogram_ends = self.get_ideogram_ends()
        self.ribbon_ends = self.get_ribbon_ends()
        self.gradients = self.get_gradients()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.ideogram_ends = self.get_ideogram_ends()
        self.ribbon_ends = self.get_ribbon_ends()
        self.gradients = self.get_gradients()

    @property
    def gradient_style(self):
        return self._gradient_style

    @gradient_style.setter
    def gradient_style(self, value):
        self._gradient_style = value
        self.gradients = self.get_gradients()

    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        self._radius = value
        self.plot_area = self.get_plot_area()

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value
        self.plot_area = self.get_plot_area()
    
    def get_plot_area(self):
        x = (-self.radius-self.padding) * 1.2
        y = x
        w = 2.4*(self.radius+self.padding) 
        h = w
        return {"x":x, "y":y, "w":w, "h":h}
    
    def draw_legend(self, fig):
        """Draw a legend in the top left showing continents and their colors"""
        if not self.continent_colors:
            return
        
        # Legend positioning (top left corner) - fix positioning
        legend_x = self.plot_area["x"] + 20
        legend_y = self.plot_area["y"] + 10  # Changed to place at top
        
        # Legend styling - shrink by 20%
        rect_width = int(12 * 0.8)  # 12 → 9.6 → 10
        rect_height = int(10 * 0.8)  # 10 → 8
        text_offset = int(22 * 0.8)  # 20 → 16
        line_height = int(16 * 0.8)  # 16 → 12.8 → 13
        
        # Get unique continents that are actually used
        used_continents = set()
        for label in self.labels:
            continent = self.country_to_continent.get(label, "Unknown")
            if continent in self.continent_colors:
                used_continents.add(continent)
        
        # Legend background - shrink by 20%
        legend_width = max(int(120 * 0.8), max(len(continent) for continent in used_continents) * int(6 * 0.8) + int(35 * 0.8)) if used_continents else int(140 * 0.8)
        legend_height = len(used_continents) * line_height + int(25 * 0.8) if used_continents else int(40 * 0.8)
        
        # Semi-transparent background for legend
        # fig.append(dw.Rectangle(
            # legend_x - 20, legend_y - 10, 
            # legend_width, legend_height,
            # fill="white", fill_opacity=0.9, 
            # stroke="#cccccc", stroke_width=1
        # ))
        
        # Legend title - shrink font by 20%
        fig.append(dw.Text(
            "Continents", 
            font_size=int(self.font_size * 0.8), 
            x=legend_x - 15, 
            y=legend_y + 5,  # Closer to top
            text_anchor="start", 
            dominant_baseline="middle",
            font_family=self.font_family,
            font_weight="bold"
        ))
        
        # Draw legend items for used continents only
        for i, continent in enumerate(sorted(used_continents)):
            y_pos = legend_y + int(20 * 0.8) + i * line_height  # Shrink start position by 20%
            
            # Color rectangle - smaller size
            fig.append(dw.Circle(
                legend_x - 15 + rect_width/2, y_pos - 0.5, 
                rect_width/2,
                fill=self.continent_colors[continent], 
                stroke="#333", stroke_width=0.5
            ))
            
            # Continent label - shrink font by 20%
            display_label = continent if len(continent) <= 15 else continent[:12] + "..."
            fig.append(dw.Text(
                display_label, 
                font_size=int((self.font_size - 2) * 0.8), 
                x=legend_x - 15 + text_offset, 
                y=y_pos + 3,
                text_anchor="start", 
                dominant_baseline="middle",
                font_family=self.font_family
            ))
        
    def show(self):
        fig = dw.Drawing(self.plot_area["w"], self.plot_area["h"], origin=(self.plot_area["x"], self.plot_area["y"]))
        # background
        fig.append(dw.Rectangle(self.plot_area["x"], self.plot_area["y"], self.plot_area["w"], self.plot_area["h"], fill=self.bg_color, fill_opacity=self.bg_transparancy))
        # make ideogram
        for i,v in enumerate(self.ideogram_ends):
            fig.append(arc(self.radius, v[0], v[1], color=self.get_color(i), thickness=self.arc_thickness))
        
        # make ribbons - handle diagonal and off-diagonal differently
        ribbon_idx = 0
        
        # First, draw thin rings for diagonal elements (self-dependencies)
        for i in self.pairs["diag"]:
            v = self.ribbon_ends[ribbon_idx]
            # For diagonal elements, draw a thin ring instead of a ribbon
            fig.append(thin_ring(
                self.radius * (1.0 - self.ribbon_gap), 
                v[0], v[1], 
                color=self.gradients[ribbon_idx], 
                opacity=0.6,
                ring_width=self.ring_width
            ))
            ribbon_idx += 1
        
        # Then, draw regular ribbons for off-diagonal elements
        for i,j in self.pairs["upper"]:
            v = self.ribbon_ends[ribbon_idx]
            fig.append(ribbon(
                self.radius * (1.0 - self.ribbon_gap), 
                v[0], v[1], v[2], v[3], 
                color=self.gradients[ribbon_idx], 
                control_strength=self.ribbon_stiffness
            ))
            ribbon_idx += 1
        
        # draw labels
        for i,v in enumerate(self.labels):
            midpoint = np.mean(self.ideogram_ends[i,:])
            x, y = pol2cart(self.radius, midpoint)
            r = norm(x, y) * (1.0+self.text_position)
            angle = midpoint * 180.0/np.pi
            if midpoint > 0.5*np.pi and midpoint < 1.5*np.pi:
                r *= -1
                angle = -(180.0-angle)
                anchor = "end"
            else:
                anchor = "start"
            label = v[:10] + "." if len(v) > 10 else v
            fig.append(dw.Text(label, font_size=self.font_size, x=r, y=0, text_anchor=anchor, dominant_baseline='middle', transform=f"rotate(%f)"%(-angle), font_family=self.font_family))
        
        # Add legend in upper left
        self.draw_legend(fig)
        
        return fig
    
    def get_ideogram_ends(self):
        arc_lens = self.row_sum * self.conversion_rate
        ideogram_ends = []
        left = self.rotation
        for arc_len in arc_lens:
            right = left + arc_len
            ideogram_ends.append([left, right])
            left = right + self.gap_size
        return np.array(ideogram_ends)

    def get_pairs(self):
        n = self.shape
        upper = []
        for i in range(n):
            for j in range(i+1, n):
                if self.matrix[i,j] != 0:
                    upper.append([i,j])
        diag = []
        for i in range(n):
            if self.matrix[i,i] != 0:
                diag.append(i)
        lower = []
        for i in range(n):
            for j in range(i):
                if self.matrix[i,j] != 0:
                    lower.append([i,j])
        pairs = {"upper": upper, "diag":diag, "lower":lower}
        return pairs
    
    def get_ribbon_ends(self):
        n = self.shape
        arc_lens = self.matrix * self.conversion_rate
        regions = np.zeros((n, n+1))
        for i in range(n):
            regions[i,0] = self.ideogram_ends[i,0]
            regions[i,1:n+1] = self.ideogram_ends[i,0] + np.cumsum(np.roll(arc_lens[i,::-1], i+1))
        ribbon_ends = []
        for i in self.pairs["diag"]:
            ribbon_ends.append([regions[i,0], regions[i,1], regions[i,0], regions[i,1]]) 
        for i,j in self.pairs["upper"]:
            k = n-j+i
            l = j-i
            ribbon_ends.append([regions[i,k], regions[i,k+1], regions[j,l], regions[j,l+1]])
        return ribbon_ends
    
    def get_gradients(self):
        gradients = []
        idx = 0
        # diagonal terms
        n = self.shape
        for i in self.pairs["diag"]:
            gradients.append(self.get_color(i))
            idx += 1
        
        # For upper triangular pairs, use color of the country with stronger dependency
        for i,j in self.pairs["upper"]:
            # Compare matrix values: matrix[i,j] vs matrix[j,i]
            value_ij = self.matrix[i,j] / self.scale[i]
            value_ji = self.matrix[j,i] / self.scale[j]
            
            if value_ij > value_ji:
                # Country i has stronger dependency on j, use country i's color
                ribbon_color = self.get_color(j)
            else:
                # Country j has stronger dependency on i, use country j's color  
                ribbon_color = self.get_color(i)
            
            gradients.append(ribbon_color)
            idx += 1
        return gradients
    
    def save_svg(self, filename):
        fig = self.show()
        fig.save_svg(filename)

    def save_png(self, filename):
        fig = self.show()
        fig.save_png(filename)
    
    def get_color(self, i):
        # If using continent-based coloring and we have labels
        if hasattr(self, 'use_continent_colors') and self.use_continent_colors and i < len(self.labels):
            return self.get_continent_color(self.labels[i])
        
        # Default behavior - cycle through colormap
        n = len(self.colormap)
        return self.colormap[i % n]
    
    def get_continent_color(self, country_name):
        """Get color based on country's continent"""
        continent = self.country_to_continent.get(country_name, "Unknown")
        return self.continent_colors.get(continent, "#808080")  # Gray for unknown countries
    
    def enable_continent_colors(self):
        """Enable continent-based coloring"""
        self.use_continent_colors = True
        self.gradients = self.get_gradients()  # Refresh gradients
    
    def disable_continent_colors(self):
        """Disable continent-based coloring (use default colormap)"""
        self.use_continent_colors = False
        self.gradients = self.get_gradients()  # Refresh gradients

    def show_colormap(self):
        swatch = dw.Drawing(66*len(self.colormap), 60)
        for i,col in enumerate(self.colormap):
            swatch.append(dw.Rectangle(i*(66), 0, 60, 60, fill=self.colormap[i]))
        return swatch