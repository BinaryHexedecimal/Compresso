// ---Register the .NET reference (Blazor sends it once) ---
window.setDotNetRefForGraph = function(dotNetRef) {
    window.DotNetRefForGraph = dotNetRef;
    console.log(" .NET reference registered for Plotly interop");
};



// ---Main render function ---
window.ensurePlotlyReadyAndRender = async (divRef, figJson, label) => { 
    
    let div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    let start = Date.now();
    while ((!div || !window.DotNetRefForGraph) && Date.now() - start < 1000) {
        console.log("⏳ Waiting for div or .NET ref...");
        await new Promise(r => setTimeout(r, 100));
        div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    }


    const fig = JSON.parse(figJson); 

    const plot = await Plotly.newPlot(
        div, fig.data, fig.layout, 
        { responsive: true, displayModeBar: true, displaylogo: false, });        


    plot.on('plotly_click', function(data) 
    { 
        console.log("plotly_click event fired!", data);
        const point = data.points[0]; 
        const nodeIndex = point?.customdata; 
        const x = point?.x; 
        const y = point?.y; 

        // Detect center node by trace name 
        const isCenterNode = point?.data?.name === 'Highlighted Nodes'; 

        if (!isCenterNode) return; 
        // Highlight edges connected to that center 
        highlightEdges(plot, nodeIndex); 

        if (window.DotNetRefForGraph && nodeIndex !== undefined) 
            { window.DotNetRefForGraph.invokeMethodAsync( 
                    "OnCenterNodeClicked", nodeIndex, x, y, label 
                ).catch(err => console.error("Failed OnCenterNodeClicked:", err));
            } 
    }); 

    plot.on('plotly_hover', function(data) 
    { 
        console.log("plotly_hover event fired!", data);
        const point = data.points[0]; 
        const nodeIndex = point?.customdata; 
        const x = point?.x; 
        const y = point?.y; 
        const isCenterNode = point?.marker?.symbol === 'x'; 
        if (isCenterNode) return; 

        // skip center nodes on hover 
        if (window.DotNetRefForGraph && nodeIndex !== undefined) 
            { window.DotNetRefForGraph.invokeMethodAsync( 
                "OnNodeHovered", nodeIndex, x, y, label 
            ).catch(err => console.error("Failed OnNodeHovered:", err)); } 
    }); 


    plot.on('plotly_unhover', function() 
    { window.clearNodeImage(div, "hoverImage"); });

};





// === Persistent highlight for clicked center node ===
window.highlightCenterNode = function(divRef, connectedEdges, x, y, imageDataUrl) {

    const div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);

    console.log(div._ev); 

    console.log(window.getEventListeners(div)); 

    if (!div || !window.Plotly) return;

    // Reset previous highlights
    if (window.lastHighlightedEdges) {
        Plotly.restyle(div, {
            'line.color': [edgeColor],
            'line.width': [edgeWidth]
        }, [0]);

    }

    // Highlight edges connected to this center
    const edgeColor = [];
    const edgeWidth = [];
    const edgeCount = div.data[0].x.filter(x => x !== null).length / 2;

    for (let i = 0; i < edgeCount; i++) {
        edgeColor.push('#888');
        edgeWidth.push(0.3);
    }

    connectedEdges.forEach(edgeIndex => {
        edgeColor[edgeIndex] = '#ff0000';
        edgeWidth[edgeIndex] = 1.5;
    });

    Plotly.restyle(div, {
        'line.color': [edgeColor],
        'line.width': [edgeWidth]
    }, [0]);

    // Show the image above the center node
    Plotly.relayout(div, {
        annotations: [{
            x: x,
            y: y,
            xref: 'x',
            yref: 'y',
            showarrow: true,
            arrowhead: 2,
            ax: 0,
            ay: -40,
            text: `<img src="${imageDataUrl}" width="90" height="90" />`
        }]
    });

    window.lastHighlightedEdges = connectedEdges;
};

// ===Temporary hover image (no highlight) ===
window.showHoverImage = function(divRef, imageDataUrl, x, y) {
    const div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    if (!div || !window.Plotly) return;

    const hoverAnnotation = {
        annotations: [{
            x: x,
            y: y,
            xref: 'x',
            yref: 'y',
            showarrow: false,
            text: `<img src="${imageDataUrl}" width="90" height="90" />`,
            name: 'hoverImage'
        }]
    };
    Plotly.relayout(div, hoverAnnotation);
};


// === Clear only hover image (keep center highlight) ===
window.clearHoverImage = function(divRef) {
    const div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    if (!div || !window.Plotly) return;

    // Remove hover-only annotations (keep others)
    const remaining = (div.layout.annotations || []).filter(a => a.name !== 'hoverImage');
    Plotly.relayout(div, { annotations: remaining });
};



// === Helper: add image overlay ===
window.addNodeImage = function (divRef, imageDataUrl, x, y, tag = "default") {

    console.log("addNodeImage called", { imageDataUrl, x, y, tag });

    const div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    if (!div || !window.Plotly) return;

    const currentLayout = div.layout || {};
    const newImage = {
        source: imageDataUrl,
        x: x,
        y: y,
        xref: "x",
        yref: "y",
        sizex: 0.15,
        sizey: 0.15,
        xanchor: "center",
        yanchor: "middle",
        layer: "above",
        name: tag, // so we can later remove specific ones
    };
    console.log(newImage);
    const updatedImages = [
        ...(currentLayout.images || []).filter(img => img.name !== tag),
        newImage
    ];

    Plotly.relayout(div, { images: updatedImages });
};

// === Helper: clear images by tag ===
window.clearNodeImage = function (divRef, tag = "default") {
    const div = divRef instanceof HTMLElement ? divRef : document.getElementById(divRef);
    if (!div || !window.Plotly) return;
    const remaining = (div.layout.images || []).filter(img => img.name !== tag);
    Plotly.relayout(div, { images: remaining });
};




async function highlightEdges(div, nodeIndex) {
    if (!window.Plotly || !div) return;

    const updateIndices = [];
    const colors = [];
    const widths = [];

    // Prepare updates in bulk
    div.data.forEach((trace, i) => {
        if (trace.name === 'edge') {
            const [u, v] = trace.customdata[0];
            if (u === nodeIndex || v === nodeIndex) {
                colors.push('#ff0000');
                widths.push(2);
            } else {
                colors.push('#888');
                 widths.push(0.5);
                //widths.push(trace.line.width);
            }
            updateIndices.push(i);
        }
    });

    // Single batch update
    await Plotly.restyle(div, {
        'line.color': colors,
        'line.width': widths
    }, updateIndices);
}
