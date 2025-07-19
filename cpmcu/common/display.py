#!/usr/bin/env python3
"""
CPM.cu Display System - Single Display Object

Single Display object containing all display functionality.
Clean and simple API with display.method() pattern.
"""

import platform
import torch
import time
from rich.console import Console, Group
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.live import Live


class Display:
    """Single display object with delayed initialization singleton"""
    
    _instance = None
    _mode_configured = False
    _use_plain_mode = False
    
    @classmethod
    def configure(cls, use_plain_mode=False):
        """Configure display mode before first use"""
        if cls._instance is not None:
            raise RuntimeError("Display already initialized, cannot configure")
        cls._use_plain_mode = use_plain_mode
        cls._mode_configured = True
    
    def __new__(cls):
        if cls._instance is None:
            if not cls._mode_configured:
                # Use default configuration if not explicitly configured
                cls._use_plain_mode = False
                cls._mode_configured = True
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance
    
    def _init_once(self):
        """Initialize display instance once"""
        self.use_plain_mode = self._use_plain_mode
        self.theme = Theme({
            "success": "bold green",
            "warning": "bold yellow", 
            "error": "bold red",
            "info": "cyan",
            "dim": "dim white",
        })
        self.console = None if self.use_plain_mode else Console(theme=self.theme)
    
    def render_line(self, text, style=None):
        """Render single line text"""
        if self.use_plain_mode:
            print(text)
        else:
            if self.console:
                self.console.print(text, style=style)

    def render_config(self, args, title="Configuration"):
        """Render configuration summary"""
        sys_info = [("OS", f"{platform.system()} {platform.release()}"), ("Python", platform.python_version())]
        if torch and torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            sys_info.extend([("GPU", f"{gpu} ({mem}GB)"), ("CUDA", torch.version.cuda), ("PyTorch", torch.__version__.split('+')[0])])
        else:
            sys_info.append(("GPU", "Not Available"))
        
        sections = [("System Information", sys_info, "yellow")]
        
        # Process configuration sections
        for section_name, config in self.CONFIG_SECTIONS.items():
            if config.get("condition") and not config["condition"](args):
                continue
                
            items = []
            for field_name, display_name, transformer in config["fields"]:
                if hasattr(args, field_name):
                    value = getattr(args, field_name)
                    if transformer:
                        value = transformer(value)
                    items.append((display_name, value))
            
            if items:
                sections.append((section_name, items, config["color"]))
        
        self._render_sections(sections, title)
    
    def render_performance(self, stats):
        """Render performance summary"""
        rows = []
        
        if 'prefill_length' in stats:
            rows.append(("Prefill Length", f"{stats['prefill_length']}", "tokens"))
        if 'prefill_time' in stats and stats['prefill_time'] > 0:
            rows.append(("Prefill Time", f"{stats['prefill_time']:.2f}", "s"))
            rows.append(("Prefill Speed", f"{stats['prefill_length'] / stats['prefill_time']:.1f}", "tokens/s"))
        
        if 'accept_lengths' in stats and stats['accept_lengths']:
            mean_accept = sum(stats['accept_lengths']) / len(stats['accept_lengths'])
            rows.append(("Mean Accept Length", f"{mean_accept:.2f}", "tokens"))
        
        if 'decode_length' in stats:
            rows.append(("Decode Length", f"{stats['decode_length']}", "tokens"))
        if 'decode_time' in stats and stats['decode_time'] > 0:
            rows.append(("Decode Time", f"{stats['decode_time']:.2f}", "s"))
            rows.append(("Decode Speed", f"{stats['decode_length'] / stats['decode_time']:.1f}", "tokens/s"))
        
        self._render_summary_table(rows, "Performance Summary", "green", show_units=True)
    
    def render_dataset_summary(self, dataset_type, model_name, total_questions, successful_questions, success_rate, summary_stats):
        """Render dataset evaluation summary"""
        rows = []
        
        # Basic information
        rows.append(("Dataset Type", dataset_type, ""))
        rows.append(("Model Name", model_name, ""))
        rows.append(("Total Questions", str(total_questions), ""))
        rows.append(("Successful Questions", str(successful_questions), ""))
        rows.append(("Success Rate", f"{success_rate:.1%}", ""))
        
        # Performance statistics
        if 'total_time' in summary_stats:
            rows.append(("Total Time", f"{summary_stats['total_time']:.2f}", "s"))
        if 'avg_time_per_question' in summary_stats:
            rows.append(("Avg Time per Question", f"{summary_stats['avg_time_per_question']:.2f}", "s"))
        if 'total_output_tokens' in summary_stats:
            rows.append(("Total Output Tokens", str(summary_stats['total_output_tokens']), ""))
        if 'avg_tokens_per_question' in summary_stats:
            rows.append(("Avg Tokens per Question", f"{summary_stats['avg_tokens_per_question']:.1f}", ""))
        if 'mean_accept_length' in summary_stats:
            rows.append(("Mean Accept Length", f"{summary_stats['mean_accept_length']:.2f}", ""))
        if 'throughput_tokens_per_sec' in summary_stats:
            rows.append(("Throughput", f"{summary_stats['throughput_tokens_per_sec']:.1f}", "tokens/s"))
        
        self._render_summary_table(rows, "Dataset Evaluation Summary", "green", show_units=True)
    
    def create_stream(self, title="Generated Response"):
        """Create streaming text display"""
        return DisplayStream(self, title)
        
    def create_progress(self, total_tokens):
        """Create progress display"""
        return DisplayProgress(self, total_tokens)

    def _format_value(self, value):
        """Format display values"""
        if isinstance(value, bool):
            return f"[green]✓[/green]" if value else f"[red]✗[/red]" if not self.use_plain_mode else ("✓" if value else "✗")
        elif isinstance(value, float):
            return f"[cyan]{value:.2f}[/cyan]" if not self.use_plain_mode else f"{value:.2f}"
        elif isinstance(value, int):
            return f"[cyan]{value}[/cyan]" if not self.use_plain_mode else str(value)
        else:
            s = str(value)
            truncated = s[:45] + '...' if len(s) > 50 else s
            return f"[cyan]{truncated}[/cyan]" if not self.use_plain_mode else truncated
            
    def _create_table(self, items, color="green", show_units=False):
        """Create table display"""
        if self.use_plain_mode:
            return items
            
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold white", min_width=18)
        table.add_column(style=color if show_units else "cyan", justify="right")
        if show_units:
            table.add_column(style="dim white")
        
        for item in items:
            if show_units and len(item) == 3:
                key, value, unit = item
                table.add_row(f"{key}:", str(value), unit or "")
            else:
                key, value = item[:2]
                formatted_value = self._format_value(value) if not show_units else str(value)
                table.add_row(f"{key}:", formatted_value)
        
        return table
        
    def _create_panel(self, content, title, color="green"):
        """Create panel container"""
        if self.use_plain_mode:
            return None
        return Panel(content, title=f"[bold {color}]{title}[/bold {color}]", border_style=color)

    def _render_sections(self, sections, title):
        """Render multiple configuration sections"""
        if self.use_plain_mode:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}")
            for section_title, items, _ in sections:
                # Inline _render_plain_section
                print(f"\n--- {section_title} ---")
                for key, value in items:
                    print(f"{key}: {value}")
            print(f"\n{'='*60}\n")
        else:
            section_panels = []
            for section_title, items, color in sections:
                table = self._create_table(items, color)
                panel = self._create_panel(table, section_title, color)
                if panel:
                    section_panels.append(panel)
            
            content_group = Group(*section_panels)
            main_panel = Panel(content_group, title=f"[bold blue]{title}[/bold blue]", border_style="blue", padding=(1, 2))
            if self.console:
                self.console.print(main_panel)
                self.console.print()
    
    def _render_summary_table(self, rows, title, color, show_units=False):
        """Render summary table"""
        if self.use_plain_mode:
            print(f"\n--- {title} ---")
            for row in rows:
                if show_units and len(row) == 3:
                    key, value, unit = row
                    print(f"{key}: {value} {unit}")
                else:
                    key, value = row[:2]
                    print(f"{key}: {value}")
        else:
            table = self._create_table(rows, color, show_units)
            panel = self._create_panel(table, title, color)
            if self.console:
                self.console.print(panel)
                self.console.print()

    CONFIG_SECTIONS = {
        "Model Configuration": {
            "color": "bright_blue",
            "fields": [
                ("model_path", "Model Path", None),
                ("model_type", "Model Type", None),
                ("dtype", "Data Type", None),
                ("draft_model_path", "Draft Model", None),
                ("frspec_path", "FRSpec Path", None),
                ("minicpm4_yarn", "MiniCPM4 YARN", None),
            ]
        },
        "System Configuration": {
            "color": "bright_blue",
            "fields": [
                ("cuda_graph", "CUDA Graph", None),
                ("memory_limit", "Memory Limit", None),
                ("chunk_length", "Chunk Length", None),
                ("plain_output", "Plain Output", None),
            ]
        },
        "Speculative Decoding": {
            "color": "bright_magenta",
            "fields": [
                ("spec_window_size", "Window Size", None),
                ("spec_num_iter", "Iterations", None),
                ("spec_topk_per_iter", "Top-K per Iter", None),
                ("spec_tree_size", "Tree Size", None),
                ("frspec_vocab_size", "FRSpec Vocab Size", None),
            ],
            "condition": lambda args: hasattr(args, 'draft_model_path') and args.draft_model_path
        },
        "Sparse Attention": {
            "color": "purple",
            "fields": [
                ("sink_window_size", "Sink Window Size", None),
                ("block_window_size", "Block Window Size", None),
                ("sparse_topk_k", "Sparse Top-K", None),
                ("sparse_switch", "Sparse Switch", None),
                ("use_compress_lse", "Use Compress LSE", None),
            ],
            "condition": lambda args: hasattr(args, 'model_type') and args.model_type == 'minicpm4'
        },
        "Server Configuration": {
            "color": "bright_green",
            "fields": [("host", "Host", None), ("port", "Port", None)],
            "condition": lambda args: hasattr(args, 'host') or hasattr(args, 'port')
        },
        "Prompt Configuration": {
            "color": "bright_green", 
            "fields": [
                ("prompt_file", "Prompt File", lambda x: bool(x)),
                ("prompt_text", "Prompt Text", lambda x: bool(x)),
                ("use_chat_template", "Use Chat Template", None),
                ("dataset", "Dataset Type", None),
                ("dataset_path", "Dataset Path", lambda x: x if x else "default"),
            ],
            "condition": lambda args: any(hasattr(args, f) for f in ['prompt_file', 'prompt_text', 'use_chat_template', 'dataset'])
        },
        "Generation Configuration": {
            "color": "bright_cyan",
            "fields": [
                ("num_generate", "Max Tokens", None),
                ("use_stream", "Use Stream", None),
                ("ignore_eos", "Ignore EOS", None),
                ("temperature", "Temperature", None),
                ("random_seed", "Random Seed", None),
            ],
            "condition": lambda args: any(hasattr(args, f) for f in ['num_generate', 'use_stream', 'ignore_eos', 'temperature', 'random_seed'])
        }
    }


class DisplayStream:
    """Streaming text display"""
    
    def __init__(self, display, title="Generated Response"):
        self.display = display
        self.title = title
        self.current_text = ""
        self.active = False
        
    def __enter__(self):
        self._show_header()
        self.active = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            self._show_footer()
            
    def append(self, text):
        """Append new text to streaming output"""
        if text is not None and self.active:
            self.current_text += text
            print(text, end="", flush=True)
        
    def replace(self, text):
        """Replace current text content"""
        if self.active:
            self.current_text = text if text is not None else ""
            print(self.current_text, end="", flush=True)
    
    def _show_header(self):
        """Show streaming output header"""
        self.display.render_line(f"\n{self.title}", "bold green")
        self.display.render_line("-" * 50)
    
    def _show_footer(self):
        """Show streaming output footer"""
        self.display.render_line("\n" + "-" * 50 + "\n")


class DisplayProgress:
    """Progress display"""
    
    def __init__(self, display, total_tokens):
        self.display = display
        self.total_tokens = total_tokens
        self.start_time = None
        self.live_display = None
        self.progress_bar = None
        self.task_id = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live_display:
            self.live_display.stop()
            
    def begin(self):
        """Start progress display"""
        if self.display.use_plain_mode:
            print("Prefilling: 0.0% (0/{} tokens) @ estimated 0.0 tokens/s".format(self.total_tokens), end="", flush=True)
        else:
            progress = self._create_progress_bar()
            if progress:
                self.live_display = Live(progress, refresh_per_second=10)
                self.live_display.start()
                
        self.start_time = time.time()
        
    def advance(self, current_tokens):
        """Advance progress display"""
        if self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        progress_percent = (current_tokens * 100.0) / self.total_tokens
        tokens_per_sec = current_tokens / elapsed_time if elapsed_time > 0 else 0.0
        
        if self.display.use_plain_mode:
            print(f"\rPrefilling: {progress_percent:.1f}% ({current_tokens}/{self.total_tokens} tokens) @ estimated {tokens_per_sec:.1f} tokens/s", end="", flush=True)
        else:
            if self.progress_bar and self.task_id is not None:
                self.progress_bar.update(
                    self.task_id,
                    completed=current_tokens,
                    speed=f"estimated {tokens_per_sec:.1f} tokens/s"
                )
                
    def finish(self):
        """Complete progress display"""
        if self.start_time is None:
            return
            
        final_elapsed_time = time.time() - self.start_time
        final_tokens_per_sec = self.total_tokens / final_elapsed_time if final_elapsed_time > 0 else 0.0
        
        if self.display.use_plain_mode:
            print(f"\rPrefilling: 100.0% ({self.total_tokens}/{self.total_tokens} tokens) @ estimated {final_tokens_per_sec:.1f} tokens/s - Complete!")
            print()
        else:
            if self.progress_bar and self.task_id is not None:
                self.progress_bar.update(
                    self.task_id,
                    completed=self.total_tokens,
                    speed=f"estimated {final_tokens_per_sec:.1f} tokens/s"
                )
                
            if self.live_display:
                self.live_display.stop()
            
        return final_elapsed_time
    
    def _create_progress_bar(self):
        """Create Rich progress bar"""
        if self.display.use_plain_mode:
            return None
            
        self.progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Prefilling"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[bold green]{task.fields[speed]}"),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=False
        )
        
        self.task_id = self.progress_bar.add_task(
            "prefill",
            total=self.total_tokens,
            speed="0.0 tokens/s"
        )
        
        return self.progress_bar

class DisplayProxy:
    """Proxy for delayed display initialization"""
    
    def __init__(self):
        self._display = None
    
    def _get_display(self):
        if self._display is None:
            self._display = Display()
        return self._display
    
    def __getattr__(self, name):
        return getattr(self._get_display(), name)


# Global display instance (proxy for delayed creation)
display = DisplayProxy()
