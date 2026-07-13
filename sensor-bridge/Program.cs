// Program.cs
//
// Polls CPU/GPU sensors using LibreHardwareMonitorLib and writes one JSON
// line per reading to live_readings.jsonl (in this same folder), on a
// timer. Python's main.py reads/tails this file during the analysis
// window.
//
// GENERALIZED: sensor matching is based on SensorType (Temperature, Clock,
// Power, Fan) plus general keyword patterns, NOT specific chip model names.
// This means it should work across AMD/Intel CPUs and AMD/Nvidia/Intel
// GPUs without editing -- as long as the vendor labels sensors with
// roughly standard terminology (which LibreHardwareMonitorLib normalizes
// to a good degree already).
//
// Run with: dotnet run
// Stop with: Ctrl+C
//
// DEBUG_MODE: set to true to print every single sensor LibreHardwareMonitor
// finds on your system, with its exact name/type/value. Useful the first
// time you run this on new hardware, to see what's actually available.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using LibreHardwareMonitor.Hardware;

class Program
{
    const int PollIntervalMs = 2000;
    const string OutputPath = "live_readings.jsonl";
    const bool DebugMode = true; // flip to false once you've confirmed real values are coming through

    // Keywords used to identify "hotspot/junction" style temperature sensors,
    // as opposed to the primary/edge temperature sensor. Covers common
    // vendor terminology without hardcoding to one specific chip.
    static readonly string[] HotspotKeywords = { "hot spot", "hotspot", "junction", "tjmax", "die" };

    static readonly HashSet<HardwareType> GpuTypes = new()
    {
        HardwareType.GpuAmd,
        HardwareType.GpuNvidia,
        HardwareType.GpuIntel,
    };

    static void Main()
    {
        Console.WriteLine("Starting sensor polling. Press Ctrl+C to stop.");
        Console.WriteLine($"Writing readings to: {Path.GetFullPath(OutputPath)}");

        var computer = new Computer
        {
            IsCpuEnabled = true,
            IsGpuEnabled = true,
        };
        computer.Open();

        using var writer = new StreamWriter(OutputPath, append: false);
        writer.AutoFlush = true;

        Console.CancelKeyPress += (sender, e) =>
        {
            Console.WriteLine("\nStopping...");
            computer.Close();
        };

        while (true)
        {
            var reading = PollOnce(computer);

            string json = JsonSerializer.Serialize(reading);
            writer.WriteLine(json);
            Console.WriteLine(json);

            Thread.Sleep(PollIntervalMs);
        }
    }

    static SensorReading PollOnce(Computer computer)
    {
        var reading = new SensorReading { Timestamp = DateTime.UtcNow.ToString("o") };

        // Collect ALL matching sensor values first, then pick sensibly
        // (e.g. max temp, average clock) -- this is what makes it
        // hardware-agnostic instead of relying on one exact sensor name.
        var cpuTemps = new List<float>();
        var cpuClocks = new List<float>();
        var cpuPowers = new List<float>();
        var gpuTemps = new List<float>();
        var gpuHotspotTemps = new List<float>();
        var gpuClocks = new List<float>();
        var gpuFans = new List<float>();

        foreach (IHardware hardware in computer.Hardware)
        {
            hardware.Update();

            foreach (ISensor sensor in hardware.Sensors)
            {
                if (sensor.Value == null) continue;
                float value = sensor.Value.Value;
                string nameLower = sensor.Name.ToLowerInvariant();

                if (DebugMode)
                {
                    Console.WriteLine($"[DEBUG] {hardware.HardwareType} | {sensor.SensorType} | {sensor.Name} = {value}");
                }

                bool isHotspotLike = HotspotKeywords.Any(k => nameLower.Contains(k));

                if (hardware.HardwareType == HardwareType.Cpu)
                {
                    switch (sensor.SensorType)
                    {
                        case SensorType.Temperature:
                            cpuTemps.Add(value);
                            break;
                        case SensorType.Clock:
                            // Skip bus/reference clocks, keep core clocks --
                            // "bus" is the one common false-positive across vendors
                            if (!nameLower.Contains("bus"))
                                cpuClocks.Add(value);
                            break;
                        case SensorType.Power:
                            cpuPowers.Add(value);
                            break;
                    }
                }
                else if (GpuTypes.Contains(hardware.HardwareType))
                {
                    switch (sensor.SensorType)
                    {
                        case SensorType.Temperature:
                            if (isHotspotLike)
                                gpuHotspotTemps.Add(value);
                            else
                                gpuTemps.Add(value);
                            break;
                        case SensorType.Clock:
                            if (!nameLower.Contains("bus"))
                                gpuClocks.Add(value);
                            break;
                        case SensorType.Fan:
                            gpuFans.Add(value);
                            break;
                    }
                }
            }
        }

        // Pick sensible aggregates:
        // - Temps: use MAX (the hottest reading is the meaningful one for
        //   thermal diagnostics, regardless of how many temp sensors exist)
        // - Clocks: use AVERAGE (matches "Core Clocks (avg)" from your
        //   training data)
        // - Power/Fan: use MAX (most representative single reading)
        reading.CpuTemp = cpuTemps.Count > 0 ? cpuTemps.Max() : (float?)null;
        reading.CpuClock = cpuClocks.Count > 0 ? cpuClocks.Average() : (float?)null;
        reading.CpuPackagePower = cpuPowers.Count > 0 ? cpuPowers.Max() : (float?)null;

        reading.GpuEdge = gpuTemps.Count > 0 ? gpuTemps.Max() : (float?)null;
        reading.GpuHotspot = gpuHotspotTemps.Count > 0 ? gpuHotspotTemps.Max()
                              : reading.GpuEdge; // fallback: some GPUs/drivers don't expose a separate hotspot sensor
        reading.GpuClock = gpuClocks.Count > 0 ? gpuClocks.Average() : (float?)null;
        reading.GpuFanRpm = gpuFans.Count > 0 ? gpuFans.Max() : (float?)null;

        return reading;
    }
}

class SensorReading
{
    public string? Timestamp { get; set; }
    public float? CpuTemp { get; set; }
    public float? CpuClock { get; set; }
    public float? CpuPackagePower { get; set; }
    public float? GpuHotspot { get; set; }
    public float? GpuEdge { get; set; }
    public float? GpuClock { get; set; }
    public float? GpuFanRpm { get; set; }
}