using CompressoApp.Components;
using CompressoApp.Services;
using CompressoApp.Models;
using Microsoft.AspNetCore.Http.Features;



var inDocker = Environment.GetEnvironmentVariable("DOTNET_RUNNING_IN_CONTAINER") == "true";

var backendUrlInternal = Environment.GetEnvironmentVariable("BACKEND_URL_INTERNAL") ?? "http://backend:8000";
var backendUrlExternal = Environment.GetEnvironmentVariable("BACKEND_URL_EXTERNAL") ?? "http://127.0.0.1:8000";

var backendUrl = inDocker ? backendUrlInternal : backendUrlExternal;


var builder = WebApplication.CreateBuilder(args);
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();



builder.Services.AddHttpClient<ApiClient>(client =>
{
    client.BaseAddress = new Uri(backendUrl);
});

builder.Services.AddSingleton(new BackendUrls
{
    Internal = backendUrlInternal,
    External = backendUrlExternal
});



builder.Services.AddScoped<ImageService>();
builder.Services.AddScoped<DatasetInfoService>();
builder.Services.AddAntiforgery(options => options.SuppressXFrameOptionsHeader = true);
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 2L * 1024L * 1024L * 1024L; // 2 GB
});




var app = builder.Build();


app.UseHttpsRedirection();

app.UseAntiforgery();

app.MapStaticAssets();

app.MapRazorComponents<App>()
   .AddInteractiveServerRenderMode();


app.Run();

