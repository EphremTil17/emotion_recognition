import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import Papa from 'papaparse';
import { 
    Box, 
    Button, 
    Typography, 
    Paper,
    Grid,
    CircularProgress,
    Divider,
    Select,
    MenuItem,
    FormControl,
    InputLabel
} from '@mui/material';

import {
    PieChart,
    Pie,
    Cell,
    Legend
} from 'recharts';

const COLORS = {
    Happy: '#05c793',    // Bright Green
    Sad: '#26547d',      // Medium Gray
    Neutral: '#fff5eb',  // Bright Blue
    Angry: '#ef436b',    // Bright Red
    Surprise: '#ffd033', // Bright Yellow   
    Undefined: '#dddddd' // Light Gray
};

const ENVIRONMENT_LABELS = {
    classroom: "Classroom",
    product_demo: "Product Demo",
    seminar: "Seminar",
    general: "General"
};

const CustomLegend = ({ payload }) => {
    return (
        <ul style={{ 
            listStyle: 'none', 
            padding: 0,
            display: 'flex',
            justifyContent: 'center',
            flexWrap: 'wrap',
            gap: '20px',
            marginTop: '20px'
        }}>
            {payload.map((entry, index) => (
                <li 
                    key={`legend-${index}`}
                    style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                        fontSize: '13px',
                        color: '#666666'
                    }}
                >
                    <span style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '3px',
                        backgroundColor: entry.color,
                        marginRight: '8px',
                        display: 'inline-block'
                    }}/>
                    {entry.value}
                </li>
            ))}
        </ul>
    );
};

const CustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    const RADIAN = Math.PI / 180;
    const radius = outerRadius * 1.2;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    const sin = Math.sin(-midAngle * RADIAN);
    const cos = Math.cos(-midAngle * RADIAN);
    const textAnchor = cos >= 0 ? 'start' : 'end';

    return percent > 0.05 ? (
        <text 
            x={x}
            y={y}
            textAnchor={textAnchor}
            dominantBaseline="central"
            fontFamily="Inter, system-ui, -apple-system, sans-serif"
            fontSize="13"
            fontWeight="500"
            fill="#666666"
        >
            {`${(percent * 100).toFixed(1)}%`}
        </text>
    ) : null;
};

const StatsItem = ({ label, value }) => (
    <Box sx={{ mb: 1.5 }}>
        <Typography variant="body2" color="text.secondary" sx={{ 
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
            fontSize: '13px'
        }}>
            {label}
        </Typography>
        <Typography variant="body1" sx={{ 
            fontWeight: 500,
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
        }}>
            {value}
        </Typography>
    </Box>
);

const ResultsView = ({ processedResults, onClose }) => {
    const [selectedEnvironment, setSelectedEnvironment] = useState('classroom');
    const [timeSeriesData, setTimeSeriesData] = useState(null);
    const [averageType, setAverageType] = useState('30s');

    useEffect(() => {
        fetch('/api/analytics', {
            method: 'GET',
            headers: {
                'Accept': 'text/csv',
            },
            credentials: 'include'
        })
        .then(response => response.text())
        .then(csv => {
            const results = Papa.parse(csv, { 
                header: true, 
                dynamicTyping: true,
                skipEmptyLines: true 
            });
            if (results.data && results.data.length > 0) {
                setTimeSeriesData(results.data);
            }
        })
        .catch(error => console.error('Error fetching analytics:', error));
    }, []);

    if (!processedResults) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
            </Box>
        );
    }

    const currentScore = processedResults.engagement_scores?.[selectedEnvironment] || 'N/A';

    const distributionData = Object.entries(processedResults.emotion_distribution)
        .filter(([name]) => name !== 'Undefined' || processedResults.emotion_distribution[name] > 0)
        .map(([name, value]) => ({ 
            name, 
            value: parseFloat(value.toFixed(1))
        }));

    const chartOptions = {
        chart: {
            type: 'area',
            height: 350,
            toolbar: {
                show: true
            },
            fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        },
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.3,
            }
        },
        xaxis: {
            type: 'numeric',
            title: {
                text: 'Time (seconds)',
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            },
            labels: {
                formatter: (value) => `${value}s`,
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            }
        },
        yaxis: {
            min: 1,
            max: 5,
            title: {
                text: 'Engagement Score',
                style: {
                    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }
            }
        },
        tooltip: {
            x: {
                formatter: (value) => `${value}s`
            },
            style: {
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
            }
        }
    };

    const getChartSeries = () => {
        if (!timeSeriesData) return [];

        const columnMap = {
            '30s': `${selectedEnvironment}_30s_avg`,
            '1min': `${selectedEnvironment}_1min_avg`,
            'cumulative': `${selectedEnvironment}_cumulative`
        };

        const column = columnMap[averageType];
        
        return [{
            name: 'Engagement Score',
            data: timeSeriesData
                .filter(row => row && row.timestamp !== undefined)
                .map(row => ({
                    x: parseFloat(row.timestamp),
                    y: parseFloat(row[column])
                }))
                .filter(point => !isNaN(point.y))
        }];
    };

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ 
                fontWeight: 300,
                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
            }}>
                Analysis Results
            </Typography>
            
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6" gutterBottom sx={{
                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                            }}>
                                Engagement Score
                            </Typography>
                            <FormControl size="small" sx={{ minWidth: 150 }}>
                                <InputLabel id="environment-select-label">Environment</InputLabel>
                                <Select
                                    labelId="environment-select-label"
                                    value={selectedEnvironment}
                                    label="Environment"
                                    onChange={(e) => setSelectedEnvironment(e.target.value)}
                                    sx={{ 
                                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                        fontSize: '14px'
                                    }}
                                >
                                    {Object.entries(ENVIRONMENT_LABELS).map(([key, label]) => (
                                        <MenuItem 
                                            key={key} 
                                            value={key}
                                            sx={{ 
                                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                                fontSize: '14px'
                                            }}
                                        >
                                            {label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Box>
                        <Typography variant="h2" color="primary" sx={{ 
                            mb: 3, 
                            fontWeight: 300,
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            {currentScore.toFixed(1)}/5.0
                        </Typography>
                        
                        <Divider sx={{ my: 2 }} />
                        
                        <Typography variant="h6" gutterBottom sx={{ 
                            mt: 2,
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            Processing Statistics
                        </Typography>
                        
                        <StatsItem 
                            label="Frames Processed"
                            value={processedResults.frames_processed}
                        />
                        <StatsItem 
                            label="Face Detection Rate"
                            value={`${processedResults.face_detection_rate.toFixed(1)}%`}
                        />
                        <StatsItem 
                            label="Duration"
                            value={`${(processedResults.duration_seconds).toFixed(1)} seconds`}
                        />
                    </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom sx={{
                            fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                        }}>
                            Emotion Distribution
                        </Typography>
                        <Box sx={{ 
                            display: 'flex', 
                            justifyContent: 'center', 
                            alignItems: 'center', 
                            height: 'calc(100% - 32px)'
                        }}>
                            <PieChart width={500} height={400}>
                                <Pie
                                    data={distributionData}
                                    dataKey="value"
                                    nameKey="name"
                                    cx="50%"
                                    cy="45%"
                                    outerRadius={130}
                                    startAngle={-270}
                                    endAngle={90}
                                    labelLine={false}
                                    label={CustomLabel}
                                >
                                    {distributionData.map((entry) => (
                                        <Cell 
                                            key={entry.name} 
                                            fill={COLORS[entry.name]}
                                            stroke="none"
                                        />
                                    ))}
                                </Pie>
                                <Legend 
                                    content={CustomLegend}
                                    verticalAlign="bottom"
                                    align="center"
                                />
                            </PieChart>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>

            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6" sx={{
                                fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                            }}>
                                Engagement Over Time
                            </Typography>
                            <FormControl size="small" sx={{ minWidth: 150 }}>
                                <InputLabel>Average Type</InputLabel>
                                <Select
                                    value={averageType}
                                    label="Average Type"
                                    onChange={(e) => setAverageType(e.target.value)}
                                    sx={{ 
                                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                                        fontSize: '14px'
                                    }}
                                >
                                    <MenuItem value="30s">30 Second Average</MenuItem>
                                    <MenuItem value="1min">1 Minute Average</MenuItem>
                                    <MenuItem value="cumulative">Cumulative Average</MenuItem>
                                </Select>
                            </FormControl>
                        </Box>
                        {timeSeriesData && (
                            <ReactApexChart
                                options={chartOptions}
                                series={getChartSeries()}
                                type="area"
                                height={350}
                            />
                        )}
                    </Paper>
                </Grid>
            </Grid>

            <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Button 
                    variant="contained" 
                    onClick={onClose}
                    sx={{ 
                        mx: 1,
                        textTransform: 'none',
                        px: 4,
                        fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
                    }}
                >
                    Close Results
                </Button>
            </Box>
        </Box>
    );
};

export default ResultsView;