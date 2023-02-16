using System;
using borderCounterApi.DTOs.BorderImage;

namespace borderCounterApi.Services
{
	public interface IBorderImageService
	{
		string GetImage(string border);
		void SaveImage(BorderImageRequest request);
	}
}

