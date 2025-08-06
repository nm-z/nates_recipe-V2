#!/usr/bin/env python3
"""
Live VNA Monitoring and Temperature Inference using hold5 model
==============================================================
Real-time script for VNA data monitoring and temperature prediction.
Uses Java VNAhl command to capture VNA data, then runs inference using hold5 model.
"""

import argparse
import time
import threading
import subprocess
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

console = Console()

class VNADataHandler(FileSystemEventHandler):
    """Handles new VNA CSV files and triggers inference."""

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.csv' and str(file_path) not in self.processed_files:
            self.processed_files.add(str(file_path))
            console.print(f"[green]New VNA data detected: {file_path.name}[/green]")

            # Wait a moment for file to be fully written
            time.sleep(1)

            # Process the VNA data
            threading.Thread(
                target=self.inference_engine.process_vna_file,
                args=(file_path,),
                daemon=True
            ).start()

class LiveVnaInference:
    """Live VNA monitoring and temperature inference using hold5 model."""

    def __init__(self):
        # Use hold5 model directory
        self.model_dir = Path("best_model_hold5")
        
        # VNA Live Data monitoring
        self.vna_data_path = Path("vna-live")
        self.vna_data_path.mkdir(parents=True, exist_ok=True)
        
        # VNAhl Java command configuration
        self.vna_jar_path = "vnaJ-hl.3.3.3_jp.jar"
        self.vna_cal_file = "NATES-miniVNA_Tiny.cal"
        
        self.live_monitoring = False
        self.last_prediction = None
        self.prediction_count = 0
        self.vna_process = None

        # Model components
        self.model = None
        self.scaler = None
        self.var_threshold = None
        self.kbest_selector = None

    def load_model_components(self):
        """Load all model components from hold5 saved artifacts."""
        if self.model is None:
            with console.status("[bold blue]Loading hold5 model components..."):
                # Load the trained model
                self.model = joblib.load(self.model_dir / "hold5_final_model.pkl")

                # Load preprocessing components
                self.scaler = joblib.load(self.model_dir / "hold5_scaler.pkl")
                self.var_threshold = joblib.load(self.model_dir / "hold5_var_threshold.pkl")
                self.kbest_selector = joblib.load(self.model_dir / "hold5_kbest_selector.pkl")

            # Create success table
            table = Table(title="Hold5 Model Components Loaded", show_header=True, header_style="bold magenta")
            table.add_column("Component", style="cyan")
            table.add_column("Type", style="green")

            table.add_row("Model", f"{type(self.model).__name__}")
            table.add_row("Scaler", f"{type(self.scaler).__name__}")
            table.add_row("Variance Threshold", f"{type(self.var_threshold).__name__}")
            table.add_row("KBest Selector", f"{type(self.kbest_selector).__name__}")

            console.print(table)

    def start_vna_capture(self):
        """Start VNAhl Java application for data capture."""
        if not os.path.exists(self.vna_jar_path):
            console.print(f"[red]Error: VNA JAR file not found at {self.vna_jar_path}[/red]")
            return False
            
        if not os.path.exists(self.vna_cal_file):
            console.print(f"[red]Error: VNA calibration file not found: {self.vna_cal_file}[/red]")
            return False

        try:
            # Start VNAhl with the correct Java command and parameters
            cmd = [
                "java",
                "-Dpurejavacomm.log=false",
                "-Dpurejavacomm.debug=false",
                "-Dfstart=45000000",
                "-Dfstop=60000000",
                "-Dfsteps=10001",
                "-DdriverId=20",
                "-Dcalfile=NATES-miniVNA_Tiny.cal",
                "-Dexports=csv",
                "-DexportDirectory=vna-live",
                "-DexportFilename=live-vna{0,date,yyMMdd}_{0,time,HHmmss}",
                "-Dscanmode=REFL",
                "-DdriverPort=ttyUSB0",
                "-DkeepGeneratorOn",
                "-jar", self.vna_jar_path
            ]
            
            console.print(f"[blue]Starting VNAhl: {' '.join(cmd)}[/blue]")
            
            # Start VNA process
            self.vna_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path.cwd())
            )
            
            console.print("[green]VNAhl started successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error starting VNAhl: {e}[/red]")
            return False

    def stop_vna_capture(self):
        """Stop VNAhl Java application."""
        if self.vna_process:
            console.print("[yellow]Stopping VNAhl...[/yellow]")
            self.vna_process.terminate()
            self.vna_process.wait()
            self.vna_process = None
            console.print("[green]VNAhl stopped[/green]")

    def run_inference(self, features):
        """Run inference on the provided features using hold5 model."""
        with console.status("[bold blue]Running hold5 model inference..."):
            # Convert to numpy array and reshape for single sample
            x_sample = np.array(features).reshape(1, -1)

            # Apply preprocessing pipeline (same as training)
            # Apply variance threshold
            x_var_threshold = self.var_threshold.transform(x_sample)
            
            # Apply KBest feature selection
            x_kbest = self.kbest_selector.transform(x_var_threshold)
            
            # Apply scaler
            x_scaled = self.scaler.transform(x_kbest)
            
            # Run prediction
            prediction = self.model.predict(x_scaled)[0]
            
            return prediction

    def extract_vna_features(self, vna_df):
        """Extract features from VNA data for hold5 model - Frequency, Return Loss, Phase, and Xs."""
        try:
            # The hold5 model expects 40,004 features from 4 columns with 10,001 values each
            # Extract raw values from Frequency(Hz), Return Loss(dB), Phase(deg), and Xs
            features = []
            
            # Frequency features
            if 'Frequency(Hz)' in vna_df.columns:
                frequency = vna_df['Frequency(Hz)'].values
                features.extend(frequency.tolist())
                console.print(f"[green]Added {len(frequency)} Frequency features[/green]")
            else:
                console.print("[red]Missing Frequency(Hz) column[/red]")
                return None
            
            # Return Loss features (S11 equivalent)
            if 'Return Loss(dB)' in vna_df.columns:
                return_loss = vna_df['Return Loss(dB)'].values
                features.extend(return_loss.tolist())
                console.print(f"[green]Added {len(return_loss)} Return Loss features[/green]")
            else:
                console.print("[red]Missing Return Loss(dB) column[/red]")
                return None
            
            # Phase features
            if 'Phase(deg)' in vna_df.columns:
                phase = vna_df['Phase(deg)'].values
                features.extend(phase.tolist())
                console.print(f"[green]Added {len(phase)} Phase features[/green]")
            else:
                console.print("[red]Missing Phase(deg) column[/red]")
                return None
            
            # Xs (Reactance) features
            if 'Xs' in vna_df.columns:
                xs = vna_df['Xs'].values
                features.extend(xs.tolist())
                console.print(f"[green]Added {len(xs)} Xs (Reactance) features[/green]")
            else:
                console.print("[red]Missing Xs column[/red]")
                return None
            
            console.print(f"[green]Total features extracted: {len(features)}[/green]")
            return features
            
        except Exception as e:
            console.print(f"[red]Error extracting features: {e}[/red]")
            return None

    def process_vna_file(self, file_path):
        """Process a VNA CSV file and run inference."""
        try:
            console.print(f"[blue]Processing VNA file: {file_path.name}[/blue]")
            
            # Read VNA data
            vna_df = pd.read_csv(file_path)
            console.print(f"[green]Loaded VNA data: {vna_df.shape}[/green]")
            
            # Extract features
            features = self.extract_vna_features(vna_df)
            if features is None:
                console.print("[red]Failed to extract features[/red]")
                return
                
            console.print(f"[green]Extracted {len(features)} features[/green]")
            
            # Run inference
            prediction = self.run_inference(features)
            
            # Display results
            self.display_vna_results(file_path.name, prediction)
            
            # Save results
            self.save_vna_result(file_path.name, prediction)
            
            # Update counters
            self.last_prediction = prediction
            self.prediction_count += 1
            
        except Exception as e:
            console.print(f"[red]Error processing VNA file: {e}[/red]")

    def display_vna_results(self, filename, prediction):
        """Display inference results."""
        # Create results panel
        results_text = f"""
File: {filename}
Temperature Prediction: {prediction:.2f}°C
Model: Hold5 ExtraTreesRegressor
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        panel = Panel(
            results_text,
            title="[bold green]Hold5 Model Inference Results[/bold green]",
            border_style="green"
        )
        console.print(panel)

    def save_vna_result(self, filename, prediction):
        """Save inference results to file."""
        results_file = Path("vna_inference_results.txt")
        
        with open(results_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {filename} | {prediction:.2f}°C\n")
        
        console.print(f"[green]Results saved to {results_file}[/green]")

    def start_vna_monitoring(self):
        """Start VNA monitoring - VNAhl runs continuously."""
        console.print("[green]VNAhl is running continuously - monitoring for new files...[/green]")
        console.print("[yellow]Press Ctrl+C to stop monitoring[/yellow]")

    def start_live_monitoring(self):
        """Start live monitoring with VNAhl integration."""
        console.print("[bold blue]Starting Live VNA Monitoring with Hold5 Model[/bold blue]")
        
        # Load model components
        self.load_model_components()
        
        # Start VNAhl capture
        if not self.start_vna_capture():
            console.print("[red]Failed to start VNAhl. Exiting.[/red]")
            return
        
        # Start file monitoring
        event_handler = VNADataHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.vna_data_path), recursive=False)
        observer.start()
        
        self.live_monitoring = True
        
        # Start VNA monitoring
        self.start_vna_monitoring()
        
        console.print("[green]Live monitoring started![/green]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        
        try:
            while self.live_monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping live monitoring...[/yellow]")
            self.stop_vna_capture()
            observer.stop()
            observer.join()
            self.live_monitoring = False
            console.print("[green]Live monitoring stopped[/green]")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live VNA Monitoring with Hold5 Model")
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = LiveVnaInference()
    
    # Start live monitoring
    inference_engine.start_live_monitoring()

if __name__ == "__main__":
    main()
