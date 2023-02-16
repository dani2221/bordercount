using System;
using System.Collections.Concurrent;
using borderCounterApi.DTOs.BorderImage;

namespace borderCounterApi.Services
{
	public class BorderImageService : IBorderImageService
	{
        private ConcurrentDictionary<string, string> imageData;

        public BorderImageService()
		{
            imageData = new ConcurrentDictionary<string, string>();
        }

        public string GetImage(string border)
        {
            return imageData.GetValueOrDefault(border, "");
        }

        public void SaveImage(BorderImageRequest request)
        {
            imageData.AddOrUpdate(request.Border, request.Image, (key, old) => request.Image);
        }
    }
}

