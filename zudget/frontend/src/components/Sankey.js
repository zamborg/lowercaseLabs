import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';

const Sankey = () => {
    const [data, setData] = useState(null);
    const svgRef = useRef();

    useEffect(() => {
        axios.get('http://localhost:8000/sankey/')
            .then(response => {
                setData(response.data);
            })
            .catch(error => {
                console.error('There was an error fetching the sankey data!', error);
            });
    }, []);

    useEffect(() => {
        if (data) {
            const { nodes, links } = data;

            const width = 960;
            const height = 500;

            const svg = d3.select(svgRef.current)
                .attr('width', width)
                .attr('height', height);

            const sankeyGenerator = sankey()
                .nodeWidth(15)
                .nodePadding(10)
                .extent([[1, 1], [width - 1, height - 6]]);

            const graph = sankeyGenerator({ 
                nodes: nodes.map(d => Object.assign({}, d)),
                links: links.map(d => Object.assign({}, d))
            });

            // Render links
            svg.append('g')
                .selectAll('.link')
                .data(graph.links)
                .enter().append('path')
                .attr('class', 'link')
                .attr('d', sankeyLinkHorizontal())
                .style('stroke-width', d => Math.max(1, d.width))
                .style('fill', 'none')
                .style('stroke', '#000')
                .style('stroke-opacity', 0.2)
                .append('title')
                .text(d => `${d.source.name} → ${d.target.name}\n${d.value}`);

            // Render nodes
            const node = svg.append('g')
                .selectAll('.node')
                .data(graph.nodes)
                .enter().append('g')
                .attr('class', 'node')
                .attr('transform', d => `translate(${d.x0},${d.y0})`);

            node.append('rect')
                .attr('height', d => d.y1 - d.y0)
                .attr('width', sankeyGenerator.nodeWidth())
                .style('fill', '#4CAF50')
                .style('stroke', '#000');

            node.append('text')
                .attr('x', -6)
                .attr('y', d => (d.y1 - d.y0) / 2)
                .attr('dy', '0.35em')
                .attr('text-anchor', 'end')
                .text(d => d.name)
                .filter(d => d.x0 < width / 2)
                .attr('x', 6 + sankeyGenerator.nodeWidth())
                .attr('text-anchor', 'start');
        }
    }, [data]);

    return (
        <div>
            <h1>Sankey Diagram</h1>
            <svg ref={svgRef}></svg>
        </div>
    );
};

export default Sankey;
